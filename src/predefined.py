import torch
import torch.nn as nn

from model import SingleLinearModel, DoubleLinearModel
from train import Trainer
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
    print(f"Final Accuracy: {final_acc}, final loss: {final_loss}")


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
    print(f"Final Accuracy: {final_acc}, final loss: {final_loss}")


if __name__ == "__main__":
    print('Starting training of predefined model')
    single_linear_model()
    # double_linear_model()