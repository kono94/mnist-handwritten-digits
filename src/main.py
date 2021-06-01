import torch
import torch.nn as nn
import torch.optim as optim

from model import SingleLinearModel, DoubleLinearModel, ConvolutionalModel

from train import Trainer
import argparse
import util
from typing import Iterator
from torch.nn import Parameter

# src/main.py --model 1 --activation 2 --loss_fn 1 --optimizer 2 --lr 0.001 --batch_size 64 --hidden_size 512 --epochs 5 --filename model1Adam > models/model1Adam_output.txt
def main():
    args = parse_arguments()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Processing data in device: {device}")
    
    model = model_chooser(
        args.model, activation_chooser(args.activation), args.hidden_size
    )
    model.to(device)    
    
    trainer: Trainer = Trainer(
        model=model,
        loss_fn=loss_fn_chooser(args.loss_fn),
        optimizer=optimizer_chooser(args.optimizer, model.parameters(), args.lr),
        batch_size=args.batch_size,
        batch_size_test=args.batch_size,
        epochs=args.epochs,
        device=device
    )
    trainer.invoke_training()

    if args.filename != None:
        trainer.save_model(args.filename)

def model_chooser(
    id: int, activation_function: nn.Module, hidden_size: int
) -> nn.Module:
    if id == 1:
        return SingleLinearModel(activation_function, hidden_size)
    elif id == 2:
        return DoubleLinearModel(activation_function, hidden_size)
    elif id == 3:
        return ConvolutionalModel()
    else:
        print("invalid model id")


def activation_chooser(id: int) -> nn.Module:
    if id == 1:
        return nn.Tanh()
    elif id == 2:
        return nn.Sigmoid()
    elif id == 3:
        return nn.ReLU()
    elif id == 4:
        return nn.LeakyReLU()
    else:
        print("Invalid activation function id")


def loss_fn_chooser(id: int) -> nn.Module:
    if id == 1:
        return nn.CrossEntropyLoss()
    elif id == 2:
        return nn.MSELoss()
    elif id == 3:
        return nn.SmoothL1Loss()
    else:
        print("Invalid loss function id")


def optimizer_chooser(id: int, params: Iterator[Parameter], lr: float) -> nn.Module:
    if id == 1:
        return torch.optim.SGD(params, lr)
    elif id == 2:
        return torch.optim.Adam(params, lr)
    elif id == 3:
        return torch.optim.AdamW(params, lr)
    else:
        print("Invalid loss function id")


def parse_arguments() -> argparse.Namespace:
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--model", type=int, default=1, metavar="M", help="model type (default: 1), 1=SingleLinearModel, 2=DoubleLinearModel, 3=CNN"
    )

    parser.add_argument(
        "--activation",
        type=int,
        default=1,
        metavar="A",
        help="activation function (default: 1), 1=TanH, 2=Sigmoid, 3=ReLU, 4=LeakyReLU",
    )

    parser.add_argument(
        "--loss_fn",
        type=int,
        default=1,
        metavar="L",
        help="loss function (default: 1), 1=CrossEntropyLoss, 2=MSE, 3=SmoothL1Loss",
    )

    parser.add_argument(
        "--optimizer",
        type=int,
        default=1,
        metavar="O",
        help="Optimizer (default: 1), 1=SGD, 2=Adam, 3=AdamW",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="E",
        help="learing rate (default: 0.001)",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=512,
        metavar="H",
        help="hidden size (default: 512)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        metavar="B",
        help="batch size (default: 64)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        metavar="E",
        help="number of epochs (default: 3)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )

    parser.add_argument(
        "--filename", type=str, default=None, metavar="SF", help="name of the .pth file"
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
