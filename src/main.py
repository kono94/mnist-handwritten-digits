import torch
import torch.nn as nn
import torch.optim as optim

from model import SingleLinearModel, DoubleLinearModel

from train import Trainer
import argparse
import util

def single_linear_model():
    single_linear_model = SingleLinearModel(
        activation_function=nn.Sigmoid(), hidden_size=512
    )
    util.print_model_parameters(single_linear_model.parameters())
    t1: Trainer = Trainer(
        model=single_linear_model,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=torch.optim.SGD(single_linear_model.parameters(), lr=1e-3),
        batch_size=64,
        epochs=5,
    )

    final_acc, final_loss = t1.invoke_training()


def double_linear_model():
    double_linear_model = DoubleLinearModel(
        activation_function=nn.Sigmoid(), hidden_size=512
    )
    util.print_model_parameters(double_linear_model.parameters())
    t1: Trainer = Trainer(
        model=double_linear_model,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=torch.optim.SGD(double_linear_model.parameters(), lr=1e-3),
        batch_size=64,
        epochs=1,
    )

    final_acc, final_loss = t1.invoke_training()


def model_chooser(id: int):
    switcher = {
        1: single_linear_model,
        2: double_linear_model,
    }
    switcher.get(id, lambda: print("Invalid model id"))()


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--model", type=int, default=1, metavar="M", help="model type (default: 1)"
    )

    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    model_chooser(args.model)


if __name__ == "__main__":
    main()
