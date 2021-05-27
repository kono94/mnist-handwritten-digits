import torch
import torch.nn as nn


class SingleLinearModel(nn.Module):
    def __init__(self, activation_function: nn = nn.Tanh(), hidden_size: int = 512):
        super(SingleLinearModel, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(28 * 28, hidden_size),
            activation_function,
            nn.Linear(hidden_size, 10),
            activation_function,
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits


class DoubleLinearModel(nn.Module):
    def __init__(self, activation_function: nn = nn.Tanh(), hidden_size: int = 512):
        super(DoubleLinearModel, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(28 * 28, hidden_size),
            activation_function,
            nn.Linear(hidden_size, hidden_size),
            activation_function,
            nn.Linear(hidden_size, 10),
            activation_function,
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits
