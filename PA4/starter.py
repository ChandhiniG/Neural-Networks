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

torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_image_directory = './data/images/train/'
test_image_directory = './data/images/test/'
train_caption_directory = './data/annotations/captions_train2014.json'
test_caption_directory = './data/annotations/captions_val2014.json'

coco_train = COCO(train_caption_directory)
coco_test = COCO(test_caption_directory)

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

with open('TestImageIds.csv', 'r') as f_test:
    reader_test = csv.reader(f_test)
    test_ids = list(reader_test)
test_ids = [int(i) for i in test_ids[0]]
test_ann_ids = coco_test.getAnnIds(test_ids)

# for i in ids:
#     if len(coco_train.imgToAnns[i]) > 0: train_ids.append(i)

vocab = Vocabulary()

embed_size = 500

transform_train = transforms.Compose([transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                            ])
transform_test = transforms.Compose([transforms.ToTensor()])

train_loader = get_loader(train_image_directory,
                          train_caption_directory,
                          ids= train_ann_ids,
                          vocab= vocab,
                          transform=transform_train,
                          batch_size=2,
                          shuffle=True,
                          num_workers=10)

val_loader = get_loader(train_image_directory,
                          train_caption_directory,
                          ids= val_ann_ids,
                          vocab= vocab,
                          transform=transform_train,
                          batch_size=2,
                          shuffle=True,
                          num_workers=10)

test_loader = get_loader(test_image_directory,
                          test_caption_directory,
                          ids= test_ann_ids,
                          vocab= vocab,
                          transform=transform_test,
                          batch_size=1,
                          shuffle=True,
                          num_workers=10)

epochs = 100

end_id = vocab.word2ind['<end>']
print("end_idx:",end_id)

#instantiate the models
encoder = Encoder(embed_size)
#TODO This
embedding_size = 300
hidden_size = 512
decoder = DecoderLSTM(embedding_size, hidden_size, len(vocab),end_index=end_id)

encoder = encoder.to(device)
decoder = decoder.to(device)

criterion = nn.CrossEntropyLoss()
#assuming the last layer in the encoder is defined as self.linear 
params = list(encoder.embed.parameters()) + list(decoder.parameters())
optimizer = optim.Adam(params, lr=1e-3)



def train():
    '''
    Train the image captioning model made of a CNN + RNN/LSTM.
    '''
    losses, losses_val = [], []
    min_loss = float('inf') # Setting min_loss to a big value to compare later
    for epoch in range(epochs):
        ts = time.time()
        losses_epoch = []
        encoder.train()
        decoder.train()
        for iter, (images, captions, length) in enumerate(train_loader):
            encoder.zero_grad()
            decoder.zero_grad()
            
            images   = images.to(device)
            captions = captions.to(device)
            targets  = pack_padded_sequence(captions, length, batch_first=True).data
            
            # Feed forward through CNN encoder and RNN decoder
            image_features = encoder(images)
            output_caption = decoder(image_features, captions)
            
            # Pack padding the output from decoder so that it matches the padded targets
            output_caption = pack_padded_sequence(output_caption, length, batch_first=True).data
            
            # Calculating loss, gradients, and updating weights
            loss = criterion(output_caption, targets)
            losses_epoch.append(loss.item())
            loss.backward()
            optimizer.step()
            
            if iter % 100 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))

        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        losses.append(np.mean(np.array(losses_epoch)))
        losses_val.append(val(epoch))
        
        # Saving best model till now
        if(min_loss>losses_val[-1]):
            torch.save(encoder, 'best_model_encoder')
            torch.save(decoder, 'best_model_decoder')
            min_loss = losses_val[-1]
        
        # Saving train/val losses and encoder/decoder every 10 epochs
        if epoch%10 == 0:
            np.save("losses",np.array(losses))
            np.save("losses_val",np.array(losses_val))
            torch.save(encoder, 'inter_encoder_%d' %(epoch))
            torch.save(decoder, 'inter_decoder_%d' %(epoch))
    
    # Saving model after all epochs are over
    torch.save(encoder, 'final_model_encoder')
    torch.save(decoder, 'final_model_decoder')
    
    
    
def val(epoch):
    losses_val = []
    ts = time.time()
    for iter, (images, captions, length) in enumerate(val_loader):
        images   = images.to(device)
        captions = captions.to(device)
           
        with torch.no_grad():
            image_features = encoder(images)
            output_caption = decoder(image_features, captions)
        
        targets= pack_padded_sequence(captions, length, batch_first=True).data
        output_caption = pack_padded_sequence(output_caption, length, batch_first=True).data
        losses_val.append(criterion(output_caption, targets).item())
    loss_mean = np.mean(np.array(losses_val))
    print("Validation loss, Epoch "+str(epoch)+":"+ str(loss_mean))
    return loss_mean
  
    
    
def test():
    # TODO: Load the best model in encoder decoder
    losses_test = []
    perplexities_test = []
    ts = time.time()
    for iter, (images, captions, length) in enumerate(test_loader):
        images = images.to(device)
        captions = captions.to(device)
           
        with torch.no_grad():
            image_features = encoder(images)
            output_caption = decoder(image_features, captions)
       
        targets= pack_padded_sequence(captions, length, batch_first=True).data
        output_caption = pack_padded_sequence(output_caption, length, batch_first=True).data
        # TODO: get the sentence generated for every image and store it in a file
        loss_image = criterion(output_caption, targets).item()
        losses_test.append(loss_image)
        perplexities_test.append(np.exp(loss_image))
       
    l_mean = np.mean(np.array(losses_test))
    p_mean = np.mean(np.array(perplexities_test))
    return l_mean, p_mean
    
if __name__ =="__main__":
    train()
    #TODO : Fix test error in loss calculation
#     l, p = test()
#     print("loss mean , perplecity mean for test" , l, p)
