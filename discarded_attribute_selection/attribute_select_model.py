import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
import torchvision
import os
import pandas as pd
from tqdm import tqdm
#from skimage import io
#from skimage.transform import rescale, resize, downscale_local_mean
from torch.utils.data import (
    Dataset,
    DataLoader,
)
from device_config import *
from skorch.net import NeuralNet


class GoogleNet(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.conv2d=nn.Conv2d(3, 3, 3, stride=1)
        self.pretrained=pretrained
        self.FC=nn.Linear(1000, 500)
    def forward(self, x):
        x=self.conv2d(x)
        x=self.pretrained(x)
        x=self.FC(x)
        return x

       
#
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#model = torchvision.models.googlenet(pretrained=True)
#model.fc=nn.Sequential(
#    nn.Linear(in_features=1024,out_features=512),
#    nn.ReLU(),
#    nn.Linear(in_features=512,out_features=128),
#    nn.ReLU(),
#    nn.Linear(in_features=128,out_features=13),
#
#) 
#model.to(device);
#model
