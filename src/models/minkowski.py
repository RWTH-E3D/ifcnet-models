#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import transforms
from src.data import IFCNetPlySparse
from src.models.Trainer import Trainer
from src.models.sampler import ImbalancedDatasetSampler
from src.models.models import MinkowskiFCNN, MinkowskiCE2, ResNet18, ResNet34
import MinkowskiEngine as ME
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from pathlib import Path


def euler_angles_to_rotation_matrix(theta, random_order=False):
    R_x = np.array(
        [[1, 0, 0], [0, np.cos(theta[0]), -np.sin(theta[0])], [0, np.sin(theta[0]), np.cos(theta[0])]]
    )

    R_y = np.array(
        [[np.cos(theta[1]), 0, np.sin(theta[1])], [0, 1, 0], [-np.sin(theta[1]), 0, np.cos(theta[1])]]
    )

    R_z = np.array(
        [[np.cos(theta[2]), -np.sin(theta[2]), 0], [np.sin(theta[2]), np.cos(theta[2]), 0], [0, 0, 1]]
    )

    matrices = [R_x, R_y, R_z]
    if random_order:
        random.shuffle(matrices)
    R = np.matmul(matrices[2], np.matmul(matrices[1], matrices[0]))
    return R


class Random3AxisRotation:
    """
    Rotate pointcloud with random angles along x, y, z axis
    The angles should be given `in degrees`.
    Parameters
    -----------
    apply_rotation: bool:
        Whether to apply the rotation
    rot_x: float
        Rotation angle in degrees on x axis
    rot_y: float
        Rotation anglei n degrees on y axis
    rot_z: float
        Rotation angle in degrees on z axis
    """

    def __init__(self, apply_rotation: bool = True, rot_x: float = None, rot_y: float = None, rot_z: float = None):
        self._apply_rotation = apply_rotation
        if apply_rotation:
            if (rot_x is None) and (rot_y is None) and (rot_z is None):
                raise Exception("At least one rot_ should be defined")

        self._rot_x = np.abs(rot_x) if rot_x else 0
        self._rot_y = np.abs(rot_y) if rot_y else 0
        self._rot_z = np.abs(rot_z) if rot_z else 0

        self._degree_angles = [self._rot_x, self._rot_y, self._rot_z]

    def generate_random_rotation_matrix(self):
        thetas = np.zeros(3, dtype=np.float)
        for axis_ind, deg_angle in enumerate(self._degree_angles):
            if deg_angle > 0:
                rand_deg_angle = random.random() * 2 * deg_angle - deg_angle
                rand_radian_angle = float(rand_deg_angle * np.pi) / 180.0
                thetas[axis_ind] = rand_radian_angle
        return euler_angles_to_rotation_matrix(thetas, random_order=True)

    def __call__(self, pos):
        if self._apply_rotation:
            M = self.generate_random_rotation_matrix()
            pos = pos @ M.T
        return pos

    def __repr__(self):
        return "{}(apply_rotation={}, rot_x={}, rot_y={}, rot_z={})".format(
            self.__class__.__name__, self._apply_rotation, self._rot_x, self._rot_y, self._rot_z
        )
    

class TranslatePointCloud:

    def __call__(self, pointcloud):
        xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
        xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
            
        translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
        return translated_pointcloud


class ShufflePointCloud:

    def __call__(self, pointcloud):
        copy = pointcloud.copy()
        np.random.shuffle(copy)
        return copy

    
def criterion(pred, labels, smoothing=True):
    """Calculate cross entropy loss, apply label smoothing if needed."""

    labels = labels.contiguous().view(-1)
    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, labels.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, labels, reduction="mean")

    return loss


def _train(data_root, class_names, epochs, batch_size,
        learning_rate, weight_decay, embedding_dim, checkpoint_dir, eval_on_test=False):

    train_transform = transforms.Compose([
        Random3AxisRotation(rot_z=360),
        TranslatePointCloud(),
        ShufflePointCloud()
    ])
    
    if eval_on_test:
        train_dataset = IFCNetPlySparse(data_root, class_names, partition="train", transform=train_transform)
        val_dataset = IFCNetPlySparse(data_root, class_names, partition="test")
    else:
        train_dataset = IFCNetPlySparse(data_root, class_names, partition="train", transform=train_transform)
        val_dataset = IFCNetPlySparse(data_root, class_names, partition="train")

        np.random.seed(42)
        perm = np.random.permutation(range(len(train_dataset)))
        train_len = int(0.7 * len(train_dataset))
        train_dataset = Subset(train_dataset, perm[:train_len])
        val_dataset = Subset(val_dataset, perm[train_len:])
    
    print(f"Train Size: {len(train_dataset)}")
    print(f"Val Size: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8,
                              collate_fn=ME.utils.SparseCollation(), sampler=ImbalancedDatasetSampler(train_dataset))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8, collate_fn=ME.utils.SparseCollation())

    model = MinkowskiCE2(3, len(class_names), embedding_dim)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    trainer = Trainer(model, train_loader, val_loader, class_names,
        optimizer, criterion, checkpoint_dir, "MinkowskiPointNet")
    trainer.train(epochs)
    return model


def train_minkowski_pointnet(config, checkpoint_dir=None, data_root=None, class_names=None, eval_on_test=False):

    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    weight_decay = config["weight_decay"]
    embedding_dim = config["embedding_dim"]
    epochs = config["epochs"]

    _train(data_root, class_names, epochs,
        batch_size, learning_rate,
        weight_decay, embedding_dim, checkpoint_dir, eval_on_test=eval_on_test)
