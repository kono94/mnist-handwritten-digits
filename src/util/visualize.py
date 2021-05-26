import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.functional import Tensor

def visualize_mnist_numbers(trainset: Tensor, amount: int = 25) -> None:
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

def print_model_parameters(params) -> None:
    for param in params:
           print(type(param), param.size())
