import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from util import visualize_mnist_numbers

transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.MNIST(
    "./data/", train=True, download=True, transform=transform
)
testset = torchvision.datasets.MNIST(
    "./data", train=False, download=True, transform=transform
)

visualize_mnist_numbers(trainset, 25)