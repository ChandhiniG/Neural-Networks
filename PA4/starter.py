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
import time


use_gpu = torch.cuda.is_available()

train_image_directory = './data/images/train/'
train_caption_directory = './data/annotations/captions_train2014.json'
coco_train = COCO(train_caption_directory)

with open('TrainImageIds.csv', 'r') as f:
    reader = csv.reader(f)
    train_ids = list(reader)

train_ids = [int(i) for i in train_ids[0]]

train_ann_ids = coco_train.getAnnIds(train_ids)

with open('ValIds.csv', 'r') as f_val:
    reader_val = csv.reader(f_val)
    val_ids = list(reader_val)

val_ids = [int(i) for i in val_ids[0]]
val_ann_ids = coco_train.getAnnIds(val_ids)


# for i in ids:
#     if len(coco_train.imgToAnns[i]) > 0: train_ids.append(i)

vocab = Vocabulary()

embed_size = 500

transform = transforms.Compose([transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                            ])

train_loader = get_loader(train_image_directory,
                          train_caption_directory,
                          ids= train_ann_ids,
                          vocab= vocab,
                          transform=transform,
                          batch_size=2,
                          shuffle=True,
                          num_workers=10)

val_loader = get_loader(train_image_directory,
                          train_caption_directory,
                          ids= val_ann_ids,
                          vocab= vocab,
                          transform=transform,
                          batch_size=2,
                          shuffle=True,
                          num_workers=10)

epochs = 5
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
    losses = []
    losses_val = []
    min_loss = 100
    for epoch in range(epochs):
        ts = time.time()
        losses_epoch = []
        encoder.train()
        decoder.train()
        for iter, (images, captions, length) in enumerate(train_loader):
            encoder.zero_grad()
            decoder.zero_grad()

            if use_gpu:
                images = images.cuda()
                captions = captions.cuda()
            else:
                images = images.cpu()
                captions = captions.cpu()
            
            targets= pack_padded_sequence(captions, length, batch_first=True).data
            #forward
            image_features = encoder(images)
            output_caption = decoder(image_features, captions)
            output_caption = pack_padded_sequence(output_caption, length, batch_first=True).data
            loss = criterion(output_caption, targets)
            losses_epoch.append(loss.item())
            loss.backward()
            optimizer.step()
             # compare with val loss and save the best model
            if iter % 100 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))

        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        losses.append(np.mean(np.array(losses_epoch)))
        losses_val.append(val(epoch))
        if(min_loss>losses_val[-1]):
            torch.save(encoder, 'best_model_encoder')
            torch.save(decoder, 'best_model_decoder')
            min_loss = losses_val[-1]
        if epoch%10 == 0:
            np.save("losses",np.array(losses))
            np.save("losses_val",np.array(losses_val))
            torch.save(encoder, 'inter_encoder_%d' %(epoch))
            torch.save(decoder, 'inter_decoder_%d' %(epoch))
    
    torch.save(encoder, 'final_model_encoder')
    torch.save(decoder, 'final_model_decoder')
    
def val(epoch):
    losses_val = []
    ts = time.time()
    for iter, (images, captions, length) in enumerate(val_loader):
        if use_gpu:
            images = images.cuda()
            captions = captions.cuda()
        else:
            images = images.cpu()
            captions = captions.cpu()
           
        with torch.no_grad():
            image_features = encoder(images)
            output_caption = decoder(image_features, captions)
        
        targets= pack_padded_sequence(captions, length, batch_first=True).data
        output_caption = pack_padded_sequence(output_caption, length, batch_first=True).data
        losses_val.append(criterion(output_caption, targets).item())
    loss_mean = np.mean(np.array(losses_val))
    print("Validation loss, Epoch "+str(epoch)+":"+ str(loss_mean))
    return loss_mean
    
    
    
if __name__ =="__main__":
    train()