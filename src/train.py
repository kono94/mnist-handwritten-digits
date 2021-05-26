import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from data.dataset import trainset, testset

# my imports
import util
from model import SingleLinearModel, DoubleLinearModel

from typing import Union, Tuple


class Trainer:
    def __init__(
        self,
        model: Union[SingleLinearModel, DoubleLinearModel],
        loss_fn: Union[nn.CrossEntropyLoss],
        optimizer: optim.Optimizer,
        batch_size: int,
        epochs: int,
    ) -> None:
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Processing data in device: {self.device}")
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epochs = epochs

        # Wrap datasets into DataLoader
        self.train_dataloader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True
        )
        self.test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

        # Display image and label.
        train_features, train_labels = next(iter(self.train_dataloader))
        print(f"Feature batch shape: {train_features.size()}")
        print(f"Labels batch shape: {train_labels.size()}")

    def train_loop(self) -> None:
        self.model.train()
        size = len(self.train_dataloader.dataset)

        for batch, (X, y) in enumerate(self.train_dataloader):
            X, y = X.to(self.device), y.to(self.device)

            # Compute prediction and loss
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test_loop(self) -> Tuple[float, float]:
        self.model.eval()
        size = len(self.test_dataloader.dataset)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in self.test_dataloader:
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= size
        correct /= size
        acc = 100 * correct

        print(f"Test Error: \n Accuracy: {(acc):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return (acc, test_loss)

    def invoke_training(self) -> Tuple[float, float]:
        acc, loss = (0,0)
        for t in range(self.epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            self.train_loop()
            acc, loss = self.test_loop()

        print("Done!")
        return acc, loss

    def save_model(self, name) -> None:
        torch.save((Path(__file__) / f"../../models/{name}").resolve())
