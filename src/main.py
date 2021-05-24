import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.MNIST(
    "./data/", train=True, download=True, transform=transform
)
testset = torchvision.datasets.MNIST(
    "./data", train=False, download=True, transform=transform
)


def visualize_mnist_numbers(amount: int = 25) -> None:
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 5, amount // 5 + (amount % 5 > 0)
    for i in range(1, amount + 1):
        sample_idx = torch.randint(len(trainset), size=(1,)).item()
        img, label = trainset[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.axis("off")
        plt.title(label)
        plt.imshow(img.squeeze(), cmap="viridis")
    plt.show()


visualize_mnist_numbers(25)
