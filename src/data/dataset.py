import torchvision
import torchvision.transforms as transforms
from pathlib import Path

transform = transforms.Compose([transforms.ToTensor()])

# Load train and test datasets
path = (Path(__file__).parents[2] / "data/").resolve()
trainset = torchvision.datasets.MNIST(
    path, train=True, download=True, transform=transform
)
testset = torchvision.datasets.MNIST(
    path, train=False, download=True, transform=transform
)
