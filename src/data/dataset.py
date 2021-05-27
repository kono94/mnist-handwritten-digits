import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor()])

# Load train and test datasets
trainset = torchvision.datasets.MNIST(
    "./data/", train=True, download=True, transform=transform
)
testset = torchvision.datasets.MNIST(
    "./data", train=False, download=True, transform=transform
)
