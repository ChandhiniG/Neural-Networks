import torch
import torch.nn as nn
import numpy as np
import pandas as pd

use_gpu = torch.cuda.is_available()
device = torch.device('cuda' if use_gpu else 'cpu')

def freeze_weights(model):
    '''
    Freeze all trainable parameter weights on model
    '''
    for param in model.parameters():
        param.requires_grad_(False)
    return model

def init_weights(m):
    '''
    Initializing weight of Decoder layers only
    '''
    # Initializing weights of Deconvolution/Conv-transpose layers
    if isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.zeros_(m.bias.data)
#     # Initializing weights of Convolution layers
#     elif isinstance(m, nn.Conv2d):
#         torch.nn.init.xavier_uniform_(m.weight.data)
#         torch.nn.init.zeros_(m.bias.data)

def save_model(model, path, name):
    '''
    Saves the model's state dictionary to a .pth file
    '''
    torch.save({
        'model_state_dict': model.state_dict(),
        }, path + name + '.pth')

def load_model(model, path):
    '''
    Loads a model's state dictionary, applies it to the given model, and returns it
    :param model: Model in to which you want to load the state dictionary
    :type model: CNNImageSegmentation
    :param path: Path to the .pth file which stores the models state dictionary
    :type path: str
    '''
    assert isinstance(path, str) and len(path) > 0, 'Possible invalid path'
    
    try:
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    except FileNotFoundError:
        print('Could not find model as given path. Returning model without loading state dictionary')
    
    return model