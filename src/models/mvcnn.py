import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import logging
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import transforms
import json
from pathlib import Path
from datetime import datetime
from src.data import SingleImgDataset, MultiviewImgDataset
from src.models.models import MVCNN, SVCNN
from src.models.Trainer import Trainer


def _get_train_val_transforms(pretrained=True):
    train_transform = [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]

    val_transform = [
        transforms.ToTensor()
    ]

    if pretrained:
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_transform.append(norm)
        val_transform.append(norm)

    train_transform = transforms.Compose(train_transform)
    val_transform = transforms.Compose(val_transform)
    return train_transform, val_transform


def _get_svcnn_loaders(data_root, class_names, batch_size, pretrained=True):
    train_transform, val_transform = _get_train_val_transforms(pretrained=pretrained)
    train_dataset = SingleImgDataset(data_root, class_names, partition="train", transform=train_transform)
    val_dataset = SingleImgDataset(data_root, class_names, partition="train", transform=val_transform)
    
    np.random.seed(42)
    perm = np.random.permutation(range(len(train_dataset)))
    train_len = int(0.7 * len(train_dataset))
    train_dataset = Subset(train_dataset, perm[:train_len])
    val_dataset = Subset(val_dataset, perm[train_len:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8)
    return train_loader, val_loader


def _pretrain_single_view(data_root, class_names, batch_size,
                        learning_rate, weight_decay, log_dir, model_dir,
                        pretrained=True, cnn_name="vgg11"):
    model = SVCNN(nclasses=len(class_names), pretrained=pretrained, cnn_name=cnn_name)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_loader, val_loader = _get_svcnn_loaders(data_root, class_names, batch_size, pretrained)

    trainer = Trainer(model, train_loader, val_loader, class_names,
        optimizer, nn.CrossEntropyLoss(), log_dir, model_dir, "SVCNN")
    trainer.train(30)
    return model


def _get_mvcnn_loaders(data_root, class_names, batch_size, num_views=12, pretrained=True):
    train_transform, val_transform = _get_train_val_transforms(pretrained=pretrained)
    train_dataset = MultiviewImgDataset(data_root, class_names, num_views, partition="train", transform=train_transform)
    val_dataset = MultiviewImgDataset(data_root, class_names, num_views, partition="train", transform=val_transform)

    np.random.seed(42)
    perm = np.random.permutation(range(len(train_dataset)))
    train_len = int(0.7 * len(train_dataset))
    train_dataset = Subset(train_dataset, perm[:train_len])
    val_dataset = Subset(val_dataset, perm[train_len:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8)
    return train_loader, val_loader


def _train_multi_view(svcnn, data_root, class_names, batch_size,
                    learning_rate, weight_decay, log_dir, model_dir,
                    pretrained=True, cnn_name="vgg11", num_views=12):
    model = MVCNN(svcnn, nclasses=len(class_names), num_views=num_views)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_loader, val_loader = _get_mvcnn_loaders(data_root, class_names, batch_size, num_views=12, pretrained=pretrained)

    trainer = Trainer(model, train_loader, val_loader, class_names,
        optimizer, nn.CrossEntropyLoss(), log_dir, model_dir, "MVCNN",
        after_load_cb=lambda x: x.view(-1, *x.shape[-3:]))
    trainer.train(30)
    return model


def train_mvcnn(data_root, class_names, batch_size,
                learning_rate, weight_decay,
                log_dir, model_dir,
                pretrained=True, cnn_name="vgg11", num_views=12):

    with (log_dir/"config.json").open("w") as f:
        json.dump({
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "data_root": str(data_root),
            "pretrained": pretrained,
            "cnn_name": cnn_name,
            "num_views": num_views
        }, f)

    svcnn = _pretrain_single_view(data_root, class_names, batch_size,
                                learning_rate, weight_decay, log_dir, model_dir,
                                pretrained=pretrained, cnn_name=cnn_name)
    mvcnn = _train_multi_view(svcnn, data_root, class_names, int(batch_size/num_views),
                            learning_rate, weight_decay, log_dir, model_dir,
                            pretrained=pretrained, cnn_name=cnn_name, num_views=num_views)
    return mvcnn
