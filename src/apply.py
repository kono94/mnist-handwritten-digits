from PIL import Image
from pathlib import Path
import torch
import torchvision.transforms as transforms
import PIL.ImageOps    
import matplotlib.pyplot as plt

from model.convolutional_model import ConvolutionalModel



with torch.no_grad():  
    
    img_path: str = (Path(__file__).parents[1] / "data/9_2.png").resolve()
    pil_img = Image.open(img_path).convert('L')
    pil_img = PIL.ImageOps.invert(pil_img)
    
    model = ConvolutionalModel()
    model.load_state_dict(torch.load(f=(Path(__file__).parents[1] / "models/cnn").resolve()))
    tenso = transforms.ToTensor()(pil_img)
    
    plt.imshow(tenso.squeeze(), cmap="gist_gray")
    plt.show()
    
    model.eval()
    p = model(tenso.unsqueeze_(1))
    print(p)
    print(p.argmax(1).item())