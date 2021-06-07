# Classic MNIST-Handwritten-Numbers DL Project

## Description
Small private project to get used to the [Pytorch
](https://pytorch.org/) framework. It is meant to have somewhat of a modular structure to support testing of different models depending on input args, predefined models and varying datasets.

## Structure

- `/src` source code with all .py files. Check the information inside this folder for a closer look.

- `/models` .pth files of the trained models. Stores the final weights of model. Use torch.load() to load it back in and make predictions on custom images (see [src/apply.py](src/apply.py))

- `/documentation` some images of the dataset and wrong guesses plus ground truth

- `/data` placeholder folder for downloaded datasets. Additionally stores custom drawn numbers to test real life examples.

## Use

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
python3 src/main.py --model 3 --epochs 5 --optimizer 3 --seed 42 --filename myModel
```