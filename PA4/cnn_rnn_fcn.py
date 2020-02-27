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
# from evaluate_captions import *

def freeze_weights(model):
    '''
    Freeze all trainable parameter weights on model
    '''
    for param in model.parameters():
        param.requires_grad_(False)
    return model

class Encoder(nn.Module):
    '''
    Resnet 50 (pretrained) based CNN encoder whose input is an image of size 224x224x3 and output is a
    vector/embedding of size embed_size
    '''
    def __init__(self, embed_size):
        '''
        :param embed_size: Size of the output embedding
        :type embed_size: int
        '''
        super().__init__()
        # Encoder - Resnet50
        model_pretrained = torchvision.models.resnet50(pretrained=True)
        model_pretrained = freeze_weights(model_pretrained)
        
        # Extracting all encoding modules from the CNN
        modules = list(model_pretrained.children())[:-1]
        self.encoder = nn.Sequential(*modules) #extracting all layers except the last layer
        
        # Layer to convert output of CNN to a vector of size embed_size
        self.embed = nn.Linear(model_pretrained.fc.in_features, embed_size) 
        
    def forward(self, x):
        '''
        :param x: tensor of size 224x224x3
        :type x: torch.Tensor
        '''
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.embed(x)
#         x = self.decoder(x)      
        return x

class DecoderLSTM(nn.Module):
    '''
    LSTM Decoder class.
    '''
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, end_index=1):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(input_size = embed_size,hidden_size = hidden_size,
                            num_layers = num_layers, batch_first = True)
        
        self.last_layer = nn.Linear(hidden_size, vocab_size)
        self.end_index = end_index
        self.softmax = torch.nn.Softmax(dim=2)
    
    def forward(self, features, captions):
        captions = captions[:, :-1]
        embed = self.embedding_layer(captions)
        embed = torch.cat((features.unsqueeze(1), embed), dim = 1)
        lstm_outputs, _ = self.lstm(embed)
        out = self.last_layer(lstm_outputs)
        
        return out
    
    def generate(self,features,sample=False,max_len=20,temperature=1):
        batch_sz = features.size()[0]
        output = [[] for i in range(batch_sz)]
        cur_idx = 0
        states = None
        features = features.unsqueeze(1)
        while(len(output[0])<max_len):
            lstm_outputs,states = self.lstm(features,states)
            out = self.last_layer(lstm_outputs)
            if sample:
                out = self.softmax(out/temperature)
                m = torch.distributions.Categorical(out)
                cur_idx = m.sample()
            else:
                cur_idx = out.argmax(2)
            for i in range(batch_sz):
                output[i].append(cur_idx[i].item())
            features = self.embedding_layer(cur_idx)
        return output


class DecoderRNN(nn.Module):
    '''
    Vanilla RNN decoder.
    '''
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
        
        self.rnn = nn.RNN(input_size = embed_size,hidden_size = hidden_size,
                            num_layers = num_layers, batch_first = True)
        
        self.last_layer = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        captions = captions[:, :-1]
        embed = self.embedding_layer(captions)
        embed = torch.cat((features.unsqueeze(1), embed), dim = 1)
        rnn_outputs, _ = self.rnn(embed)
        out = self.last_layer(rnn_outputs)
        
        return out
    
    def generate(self,features,sample=False,max_len=20,temperature=1):
        batch_sz = features.size()[0]
        output = [[] for i in range(batch_sz)]
        cur_idx = 0
        states = None
        features = features.unsqueeze(1)
        while(len(output[0])<max_len):
            rnn_outputs,states = self.rnn(features,states)
            out = self.last_layer(rnn_outputs)
            if sample:
                out = self.softmax(out/temperature)
                m = torch.distributions.Categorical(out)
                cur_idx = m.sample()
            else:
                cur_idx = out.argmax(2)
            for i in range(batch_sz):
                output[i].append(cur_idx[i].item())
            features = self.embedding_layer(cur_idx)
        return output
