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
from model_CAE_2 import ConvAutoEncoder
from utils import init_weights, save_model, load_model

# Setting seed for reproducibility
seed = 420
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epochs    = 15
criterion = nn.MSELoss()
model     = ConvAutoEncoder()
model.apply(init_weights) #intiializes all weights (both conv and deconv layers)
optimizer = optim.Adam(model.parameters(), lr=2e-4)

model.to(device)
criterion.to(device)

# env = launch_env()
# print("Initialized environment")

# # Wrappers
# env = ResizeWrapper(env)
# env = NormalizeWrapper(env)
# env = ImgWrapper(env) # to make the images from 160x120x3 into 3x160x120
# env = ActionWrapper(env)
# env = DtRewardWrapper(env)
# print("Initialized Wrappers")

train_batch_size = 40
val_batch_size   = 10
num_batches      = 1000

def get_train_data(size):
    # Launch the env with our helper function
    replay_buffer = ReplayBuffer(size)
    obs = env.reset()
    done = False
    
    print('Start: Collecting train data')
    ts = time.time()
    for i in range(int(size)):
        
        if i%10==0 or done:
            obs = env.reset()
            done = False
        else:
            action = env.action_space.sample()
            obs, _, done, _ = env.step(action)
        
        replay_buffer.add(obs.astype(float), 0, 0, 0, 0)
    
    print('End: Collecting train data. Time taken: {:.2f}'.format(time.time() - ts))
    return replay_buffer

def save_data():
    '''
    Save images from environment so we can train on it later.
    '''
    for i in range(num_batches):    
        num_train_images  = int(train_batch_size)
        replay_buffer     = get_train_data(num_train_images)
        num_val_images    = int(val_batch_size)
        replay_buffer_val = get_train_data(num_val_images)
        # Saving val and train images to a compressed numpy file
        np.savez_compressed('data/compressed'+str(i), val=replay_buffer_val.storage, train=replay_buffer.storage)

def load_val_data():
    val_data = []
    ts = time.time()
    for i in range(num_batches):    
        # Loading the ith replay buffer
        batch_i = np.load('data/compressed'+str(i)+'.npz', allow_pickle=True)
        replay_buffer_val_i = batch_i['val']
        batch = [replay_buffer_val_i[i,0] for i in range(val_batch_size)]
        val_data += batch
    print(f'Validation set of {len(val_data)} number of images loaded in {time.time() - ts} seconds')
    return val_data

def train():
    model_dir = "./CAE_models_2/"
    data_dir  = "./data/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    losses      = []
    val_losses  = []
    min_loss    = float('inf')
    batch_size  = train_batch_size
    val_data    = load_val_data()
    
    for epoch in range(epochs):
        model.train()
        loss_batch = []
        shuffled_indices = list(range(num_batches))
        random.shuffle(shuffled_indices)
        ts = time.time()
        for i in shuffled_indices:
            optimizer.zero_grad()
            # Making batch
            batch_i = np.load('data/compressed'+str(i)+'.npz', allow_pickle=True)
            replay_buffer_train = batch_i['train']
            batch = [replay_buffer_train[j,0] for j in range(train_batch_size)]

#             if epoch == 0:
#                 fig, axs = plt.subplots(3,2,figsize=(9,7))
#                 axs = axs.flatten()
#                 for q in range(6):
#                     img = batch[q]
#                     img = np.transpose(img, (2, 1, 0))
#                     axs[q].imshow(img)
#                 plt.savefig('sample_batch.png', dpi=300)
#                 plt.close()
#                 print('img.shape = ', img.shape)
#                 print('np.sum(img) = ', np.sum(img))

            obs = np.stack(batch, axis=0)
            obs = torch.from_numpy(obs).float()
            obs = obs.to(device)
            # Feed forward
            output = model(obs)
            loss   = criterion(output, obs)
            loss_batch.append(loss.item())
            # Backprop
            loss.backward()
            optimizer.step()
            
            if i%100 == 0:
                print("epoch{}, batch: {}, loss: {}".format(epoch, i, loss_batch[-1]))
        
        losses.append(np.mean(np.array(loss_batch)))
        val_losses.append(val(val_data))
        print("Finish epoch {}, time elapsed {:.2f}, loss: {:.3f}, val_loss: {:.3f}".format(epoch, time.time() - ts, losses[-1], val_losses[-1]))
        
        if epoch%5==0 or val_losses[-1] < min_loss:
            save_model(model, model_dir, 'model_best')
            min_loss = val_losses[-1]
    
    np.save(model_dir + "CAE_losses_train",np.array(losses))
    np.save(model_dir + "CAE_losses_val",np.array(val_losses))
    save_model(model, model_dir, 'model_final')

def val(val_data):
    '''
    :param val_data: list of val data
    '''
    model.eval()
    
    val_batch         = []
    num_val_images    = len(val_data)
    shuffled_indices  = list(range(num_val_images))
    
    for i in range(0, num_val_images, val_batch_size):
        # Making batch
        batch_indices = shuffled_indices[i: i+val_batch_size]
        obs_batch     = [val_data[j] for j in batch_indices]
        obs           = np.stack(obs_batch, axis=0)
        obs           = torch.from_numpy(obs).float()
        obs           = obs.to(device)
        # Feed forward
        output        = model(obs)
        loss          = criterion(output, obs)
        val_batch.append(loss.item())
    
    return np.mean(np.array(val_batch))

if __name__ == '__main__':
    train()
    