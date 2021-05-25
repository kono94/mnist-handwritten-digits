import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# my imports
import util

BATCH_SIZE: int = 64
BATCH_SIZE_TEST: int = 64

transform = transforms.Compose([transforms.ToTensor()])

# Load train and test datasets
trainset = torchvision.datasets.MNIST(
    "./data/", train=True, download=True, transform=transform
)
testset = torchvision.datasets.MNIST(
    "./data", train=False, download=True, transform=transform
)

# Wrap datasets into DataLoader
train_dataloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
test_dataload = DataLoader(testset, batch_size=BATCH_SIZE_TEST, shuffle=True)

# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

util.visualize_mnist_numbers(trainset, 25)
