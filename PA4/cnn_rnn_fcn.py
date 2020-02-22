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

#LSTM decoder class
class DecoderLSTM(nn.Module):
    def __init__(self, embed_size=4, hidden_size, vocab_size, num_layers=1,embed=True):
        super().__init__()
        #Embedding layer : TODO: Check if it needed
        if embed:
            self.embedding_layer = nn.Embedding(vocab_size, embed_size)
        else:
            self.embedding_layer = OneHot(vocab_size)
        
        self.lstm = nn.LSTM(input_size = embed_size,hidden_size = hidden_size,
                            num_layers = num_layers, batch_first = True)
        
        self.last_layer = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        captions = captions[:, :-1]
        embed = self.embedding_layer(captions)
        embed = torch.cat((features.unsqueeze(1), embed), dim = 1)
        lstm_outputs, _ = self.lstm(embed)
        out = self.last_layer(lstm_outputs)
        
        return out