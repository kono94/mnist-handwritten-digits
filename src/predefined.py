import torch
import torch.nn as nn

from model import SingleLinearModel, DoubleLinearModel
from model.convolutional_model import ConvolutionalModel
from train import Trainer
import util


def single_linear_model() -> Trainer:
    single_linear_model = SingleLinearModel(
        activation_function=nn.Tanh(), hidden_size=512
    )
    util.print_model_parameters(single_linear_model.parameters())

    return Trainer(
        model=single_linear_model,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=torch.optim.AdamW(single_linear_model.parameters(), lr=1e-3),
        batch_size=64,
        batch_size_test=5000,
        epochs=6,
    )


def double_linear_model() -> Trainer:
    double_linear_model = DoubleLinearModel(
        activation_function=nn.Sigmoid(), hidden_size=512
    )
    util.print_model_parameters(double_linear_model.parameters())
    return Trainer(
        model=double_linear_model,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=torch.optim.SGD(double_linear_model.parameters(), lr=1e-3),
        batch_size=64,
        batch_size_test=64,
        epochs=1,
    )

def convolutional_model() -> Trainer:
    cnn = ConvolutionalModel()
    
    util.print_model_parameters(cnn.parameters())
    return Trainer(
        model=cnn,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=torch.optim.AdamW(cnn.parameters(), lr=1e-3),
        batch_size=64,
        batch_size_test=64,
        epochs=10
    )
    
if __name__ == "__main__":
    print("Starting training of predefined model")

    trainer: Trainer = convolutional_model()

    final_acc, final_loss = trainer.invoke_training()
    print(f"Final Accuracy: {final_acc}, final loss: {final_loss}")
    trainer.save_model("cnn")
    # double_linear_model()
