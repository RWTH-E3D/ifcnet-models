import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data.sampler import Sampler
import copy
import time
from typing import Callable


class ImbalancedDatasetSampler(Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):
        Sampler.__init__(self, dataset)

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        labels = self._get_labels(dataset)
        frequencies = np.bincount(labels)

        # weight for each sample
        weights = [1.0 / frequencies[labels[idx]]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    @staticmethod
    def _get_labels(dataset):
        labels = []
        for i in range(len(dataset)):
            coords, feats, label = dataset[i]
            labels.append(label)
        return labels

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
    