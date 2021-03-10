import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score)
from ray import tune
from pathlib import Path
import numpy as np
import pickle
import os
import time
import logging

class Trainer:

    def __init__(self, model, train_loader, val_loader, class_names, optimizer,
                loss_fn, checkpoint_dir, model_name, after_load_cb=None):

        self.optimizer = optimizer
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.class_names = class_names
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir
        self.after_load_cb = after_load_cb
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.load()

    def train(self, epochs, global_step=0):
        
        for epoch in range(global_step, epochs + global_step):
            self.model.train()
            all_probs = []
            all_labels = []
            running_loss = 0.0

            for data, labels in self.train_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                if self.after_load_cb:
                    data = self.after_load_cb(data)

                self.optimizer.zero_grad()

                outputs = self.model(data)

                loss = self.loss_fn(outputs, labels)
                running_loss += loss.item()
                
                probs = F.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().detach().numpy())
                all_labels.append(labels.cpu().numpy())

                loss.backward()
                self.optimizer.step()

            all_probs = np.concatenate(all_probs)
            all_labels = np.concatenate(all_labels)
            train_metrics = self.calc_metrics(all_probs, all_labels,
                                running_loss/len(self.train_loader.dataset), "train")
            val_metrics = self.evaluate()

            self.save(epoch)
            metrics = {**train_metrics, **val_metrics}
            tune.report(**metrics)

    def evaluate(self):
        self.model.eval()
        all_probs = []
        all_labels = []
        running_loss = 0.0

        with torch.no_grad():
            for data, labels in self.val_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                if self.after_load_cb:
                    data = self.after_load_cb(data)

                outputs = self.model(data)

                loss = self.loss_fn(outputs, labels)
                running_loss += loss.item()
                
                probs = F.softmax(outputs, dim=1)
                
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

            all_probs = np.concatenate(all_probs)
            all_labels = np.concatenate(all_labels)
            metrics = self.calc_metrics(all_probs, all_labels,
                        running_loss/len(self.val_loader.dataset), "val")
            return metrics

    def calc_metrics(self, probabilities, labels, loss, tag):
        predictions = np.argmax(probabilities, axis=1)

        acc = accuracy_score(labels, predictions)
        balanced_acc = balanced_accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average="weighted")
        recall = recall_score(labels, predictions, average="weighted")
        f1 = f1_score(labels, predictions, average="weighted")
        
        return {
            f"{tag}_loss": loss,
            f"{tag}_accuracy_score": acc,
            f"{tag}_balanced_accuracy_score": balanced_acc,
            f"{tag}_precision_score": precision,
            f"{tag}_recall_score": recall,
            f"{tag}_f1_score": f1
        }

    def save(self, epoch):
        with tune.checkpoint_dir(epoch) as d:
            target_path = Path(d) / "checkpoint"
            torch.save((self.model.state_dict(), self.optimizer.state_dict()), target_path)
        
    def load(self):
        if not self.checkpoint_dir: return

        path = self.checkpoint_dir / "checkpoint"
        if not path.exists(): return

        model_state, optimizer_state = torch.load(path)
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)
