import torch
import torch.nn as nn
import os
import numpy as np
import nltk
import csv
from data_loader import get_loader
from torchvision import transforms
import torchvision
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from build_vocab import Vocabulary
from pycocotools.coco import COCO
from cnn_rnn_fcn import *


use_gpu = torch.cuda.is_available()

train_image_directory = './data/images/train/'
train_caption_directory = './data/annotations/captions_train2014.json'
coco_train = COCO(train_caption_directory)

with open('TrainImageIds.csv', 'r') as f:
    reader = csv.reader(f)
    train_ids = list(reader)

train_ids = [int(i) for i in train_ids[0]]

train_ann_ids = coco_train.getAnnIds(train_ids)

# for i in ids:
#     if len(coco_train.imgToAnns[i]) > 0: train_ids.append(i)

vocab = Vocabulary()

embed_size = 500

transform = transforms.Compose([transforms.ToTensor()])

train_loader = get_loader(train_image_directory,
                          train_caption_directory,
                          ids= train_ann_ids,
                          vocab= vocab,
                          transform=transform,
                          batch_size=2,
                          shuffle=True,
                          num_workers=10)

epochs = 1
#instantiate the models
encoder = Encoder(embed_size)
#TODO This
decoder = DecoderLSTM(500, 256, len(vocab))
if use_gpu:
    encoder = encoder.cuda()
    decoder = decoder.cuda()

    
criterion = nn.CrossEntropyLoss()
#assuming the last layer in the encoder is defined as self.linear 
params = list(encoder.embed.parameters()) + list(decoder.parameters())
optimizer = optim.Adam(params, lr=5e-3)



def train():
    for epoch in range(epochs):
        for iter, (images, captions, length) in enumerate(train_loader):
            encoder.zero_grad()
            decoder.zero_grad()
            print(images.shape)

#             if use_gpu:
#                 images = images.cuda()
#                 captions = captions.cuda()
#             else:
#                 images = images.cpu()
#                 captions = captions.cpu()
#                 targets= pack_padded_sequence(captions, lengths, batch_first=True)[0]

#                 #forward
#                 image_features = encoder(images)
#                 output_caption = decoder(image_features, captions, length)
#                 loss = criterion(outputs, target)
#                 loss.backward()
#                 optimizer.step()

                #compare with val loss and save the best model
    
    #save the final model

    
    
if __name__ =="__main__":
    train()