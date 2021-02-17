#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import transforms
from src.data import IFCNetPly
from src.models.Trainer import Trainer
from src.models.models import DGCNN
import numpy as np
from torch.utils.data import DataLoader, Subset
from pathlib import Path


def _cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


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


def _train(data_root, class_names, epochs, batch_size,
        learning_rate, weight_decay,
        k, embedding_dim, dropout, checkpoint_dir):

    train_tranform = transforms.Compose([
        TranslatePointCloud(),
        ShufflePointCloud()
    ])

    train_dataset = IFCNetPly(data_root, class_names, partition="train", transform=train_tranform)
    val_dataset = IFCNetPly(data_root, class_names, partition="train")
    
    np.random.seed(42)
    perm = np.random.permutation(range(len(train_dataset)))
    train_len = int(0.7 * len(train_dataset))
    train_dataset = Subset(train_dataset, perm[:train_len])
    val_dataset = Subset(val_dataset, perm[train_len:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8)

    model = DGCNN(dropout, k, embedding_dim, len(class_names))

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    trainer = Trainer(model, train_loader, val_loader, class_names,
        optimizer, _cal_loss, checkpoint_dir, "DGCNN",
        after_load_cb=lambda x: x.permute(0, 2, 1))
    trainer.train(epochs)
    return model


def train_dgcnn(config, checkpoint_dir=None, data_root=None, class_names=None):

    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    weight_decay = config["weight_decay"]
    k = config["k"]
    embedding_dim = config["embedding_dim"]
    dropout = config["dropout"]
    epochs = config["epochs"]

    _train(data_root, class_names, epochs,
        batch_size, learning_rate,
        weight_decay, k, embedding_dim, dropout,
        checkpoint_dir)
