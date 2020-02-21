import sys

import torch
import torch.nn as nn
from torchvision import utils
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time

# from data_loader import *
from evaluate_captions import *

class CnnRnnModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder - Resnet50
        # TODO: Change out_features for Linear layer according to decoder architect and input image size
        model_pretrained = torchvision.models.resnet50(pretrained=True)
        # Adding trainable last layer
        model_pretrained.fc = nn.Linear(in_features=2048, out_features=500, bias=True)
        self.encoder = model_pretrained
        
        # Decoder: Defining RNN/LSTM encoder
        # TODO: self.decoder = 
        
    def forward(self, x):
        x = self.encoder(x)
#         x = self.decoder(x)
        
        return x