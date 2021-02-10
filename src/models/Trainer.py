import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score)
from tqdm import tqdm
import numpy as np
import pickle
import os
import time
import logging

class Trainer:

    def __init__(self, model, train_loader, val_loader, class_names, optimizer,
                loss_fn, log_dir, model_dir, model_name, after_load_cb=None):

        self.optimizer = optimizer
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.log_dir = log_dir
        self.class_names = class_names
        self.model_name = model_name
        self.model_dir = model_dir
        self.after_load_cb = after_load_cb
        self.scheduler = ReduceLROnPlateau(self.optimizer, verbose=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.writer = SummaryWriter(log_dir)

        logging.info(f"Training on {self.device}")

    def train(self, epochs):
        
        for epoch in range(epochs):
            self.model.train()
            all_probs = []
            all_labels = []
            running_loss = 0.0

            # train one epoch
            for data, labels in tqdm(self.train_loader):
                data, labels = data.to(self.device), labels.to(self.device)
                if self.after_load_cb:
                    data = self.after_load_cb(data)

                self.optimizer.zero_grad()

                outputs = self.model(data)

                loss = self.loss_fn(outputs, labels)
                running_loss += loss.item()
                
                probs = F.softmax(outputs, dim=0)
                all_probs.append(probs.cpu().detach().numpy())
                all_labels.append(labels.cpu().numpy())

                loss.backward()
                self.optimizer.step()

            all_probs = np.concatenate(all_probs)
            all_labels = np.concatenate(all_labels)
            self.write_statistics_to_tensorboard(all_probs, all_labels,
                running_loss/len(self.train_loader.dataset), epoch, "Training")
            metrics = self.evaluate(epoch)

            self.model.save(self.model_dir, self.model_name, epoch)
            self.scheduler.step(metrics["balanced_accuracy_score"])

        self.writer.close()

    def evaluate(self, epoch):
        self.model.eval()
        all_probs = []
        all_labels = []
        running_loss = 0.0

        with torch.no_grad():
            for data, labels in tqdm(self.val_loader):
                data, labels = data.to(self.device), labels.to(self.device)
                if self.after_load_cb:
                    data = self.after_load_cb(data)

                outputs = self.model(data)

                loss = self.loss_fn(outputs, labels)
                running_loss += loss.item()
                
                probs = F.softmax(outputs, dim=0)
                
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

            all_probs = np.concatenate(all_probs)
            all_labels = np.concatenate(all_labels)
            metrics = self.write_statistics_to_tensorboard(all_probs, all_labels,
                        running_loss/len(self.val_loader.dataset), epoch, "Validation")
            return metrics

    def write_statistics_to_tensorboard(self, probabilities, labels, loss, epoch, tag, add_pr_curves=False):
        predictions = np.argmax(probabilities, axis=1)

        acc = accuracy_score(labels, predictions)
        balanced_acc = balanced_accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average="weighted")
        recall = recall_score(labels, predictions, average="weighted")
        f1 = f1_score(labels, predictions, average="weighted")
        logging.info(f"Epoch: {epoch}")
        logging.info(f"{self.model_name} {tag} loss: {loss}")
        logging.info(f"{self.model_name} {tag} accuracy: {acc}")
        logging.info(f"{self.model_name} {tag} accuracy (balanced): {balanced_acc}")
        logging.info(f"{self.model_name} {tag} precision (weighted): {precision}")
        logging.info(f"{self.model_name} {tag} recall (weighted): {recall}")
        logging.info(f"{self.model_name} {tag} F1-Score (weighted): {f1}")

        tensorboard_metrics = {
            "loss": loss,
            "accuracy": acc,
            "accuracy (balanced)": balanced_acc,
            "precision (weighted)": precision,
            "recall (weighted)": recall,
            "F1-Score (weighted)": f1
        }

        for i, param_group in enumerate(self.optimizer.param_groups):
            tensorboard_metrics[f"learning rate (group {i})"] = param_group["lr"]

        self.writer.add_scalars(f"{self.model_name} {tag}", 
                                tensorboard_metrics, epoch)            

        if add_pr_curves:
            for class_index, class_name in enumerate(self.class_names):
                tensorboard_preds = predictions == class_index
                tensorboard_probs = probabilities[:, class_index]

                self.writer.add_pr_curve(class_name,
                                    tensorboard_preds,
                                    tensorboard_probs)
        
        return {
            "loss": loss,
            "accuracy_score": acc,
            "balanced_accuracy_score": balanced_acc,
            "precision_score": precision,
            "recall_score": recall,
            "f1_score": f1
        }
