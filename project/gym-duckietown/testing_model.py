import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import os
import time
import argparse
import ast
import logging
import datetime
import numpy as np
import random
import matplotlib.pyplot as plt

# # Duckietown Specific
# from learning.reinforcement.pytorch.ddpg import DDPG
# from learning.reinforcement.pytorch.utils import seed, evaluate_policy, ReplayBuffer
# from learning.utils.env import launch_env
# from learning.utils.wrappers import NormalizeWrapper, ImgWrapper, \
#     DtRewardWrapper, ActionWrapper, ResizeWrapper

# Model Specific
from model_CAE import ConvAutoEncoder
from utils import load_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ConvAutoEncoder()
model = load_model(model, './CAE_models/model_best.pth')
model = model.to(device)
model.eval()

# Loading a train batch to test the model's encoder on
i                   = 10
batch_size          = 40
batch_i             = np.load('data/compressed'+str(i)+'.npz', allow_pickle=True)
replay_buffer_train = batch_i['train']
batch               = [replay_buffer_train[j,0] for j in range(batch_size)]
input_batch         = np.stack(batch, axis=0)
input_batch         = torch.from_numpy(input_batch).float()
input_batch         = input_batch.to(device)

print('input_batch.shape = ', input_batch.shape)

# Feed forward
output, _, _ = model.forward_encoder(input_batch)
# Passing output of encoder through maxpool to further reduce dimensionality
maxpool      = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
output       = maxpool(output)
output       = output.view(batch_size, -1)
print('output.shape = ', output.shape)