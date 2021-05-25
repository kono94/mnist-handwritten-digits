import torch.nn as nn

class Activation:
    RELU = nn.ReLU()
    LEAKY_RELU = nn.LeakyReLU()
    SIGMOID = nn.Sigmoid()
    TANH = nn.Tanh()