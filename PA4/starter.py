import torch
import torch.nn as nn
import os
import datetime
import numpy as np
import nltk
import csv
import json
from data_loader import get_loader
from torchvision import transforms
import torchvision
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from build_vocab import Vocabulary
from pycocotools.coco import COCO
from cnn_rnn_fcn import *
import time
from generate_results import generate_captions
from evaluate_captions import evaluate_captions
from build_vocab import Vocabulary, get_glove


torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_image_directory = './data/images/train/'
test_image_directory = './data/images/test/'
train_caption_directory = './data/annotations/captions_train2014.json'
test_caption_directory = './data/annotations/captions_val2014.json'

coco_train = COCO(train_caption_directory)
coco_test = COCO(test_caption_directory)

with open('TrainIds.csv', 'r') as f:
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

###-------------- Start: Hyper parameters ----------------
learning_rate = 2e-5
batch_size = 64
epochs = 101
embed_size = 300
hidden_size = 512
extra_notes = ''

config = {
	'learning_rate': learning_rate,
	'batch_size': batch_size,
	'epochs': epochs,
	'embed_size': embed_size,
	'hidden_size': hidden_size,
	'extra_notes': extra_notes,
	}
###-------------- End: Hyper parameters ----------------

vocab = Vocabulary(2)

transform_train = transforms.Compose([
                                transforms.Resize(224),
                                transforms.RandomCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225]),
                            ])
transform_test = transforms.Compose([transforms.ToTensor()])

train_loader = get_loader(train_image_directory,
                          train_caption_directory,
                          ids= train_ann_ids,
                          vocab= vocab,
                          transform=transform_train,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=10)

val_loader = get_loader(train_image_directory,
                          train_caption_directory,
                          ids= val_ann_ids,
                          vocab= vocab,
                          transform=transform_train,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=10)

test_loader = get_loader(test_image_directory,
                          test_caption_directory,
                          ids= test_ann_ids,
                          vocab= vocab,
                          transform=transform_train,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=10)


end_id = vocab.word2ind['<end>']
print("end_idx:",end_id)

# Getting pre-trained embeddings
use_glove = False
if use_glove:
    embeddings = get_glove(vocab.word2ind, 'glove.6B.50d.txt')  
    
#instantiate the models
encoder = Encoder(embed_size)
if use_glove:
    decoder = DecoderLSTM(embed_size, hidden_size, len(vocab), end_index=end_id, embeddings = embeddings)
else:
    decoder = DecoderLSTM(embed_size, hidden_size, len(vocab), end_index=end_id)

encoder = encoder.to(device)
decoder = decoder.to(device)

criterion = nn.CrossEntropyLoss()
#assuming the last layer in the encoder is defined as self.linear 
params = list(encoder.embed.parameters()) + list(decoder.parameters())

optimizer = optim.Adam(params, lr=learning_rate,weight_decay=1e-5)
# optimizer = optim.Adam(params, lr=learning_rate)

def train():
    '''
    1. Train the image captioning model made of a CNN + RNN/LSTM.
    2. Save the train/val loss, model every 10 epochs (inter_model), best model, and final_model
    '''
    print('Config of training is: \n {}'.format(config))
    # Creating timestamped folder to store all outputs of train
    folder_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    mydir = os.path.join(os.getcwd(), folder_name)
    os.makedirs(mydir)
    # Saving config file
    with open(mydir+'/config.json', 'w') as f_out:
        json.dump(config, f_out)

    losses, losses_val = [], []
    min_loss = float('inf') # Setting min_loss to a big value to compare later
    for epoch in range(epochs):
        ts = time.time()
        losses_epoch = []
        encoder.train()
        decoder.train()
        for iter, (images, captions, length, _) in enumerate(train_loader):
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

            if iter % 300 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))
        
        losses.append(np.mean(np.array(losses_epoch)))
        losses_val.append(val(epoch))
        print("Finish epoch {}, time elapsed: {:.2f}, loss: {:.4f}".format(epoch, time.time() - ts, losses[-1]))
        
        # Saving best model till now
        if(min_loss>losses_val[-1]):
            torch.save(encoder, mydir + '/best_model_encoder')
            torch.save(decoder, mydir + '/best_model_decoder')
            min_loss = losses_val[-1]
        
        # Saving train/val losses and encoder/decoder every 10 epochs
        if epoch%10 == 0:
            np.save(mydir + "/losses",np.array(losses))
            np.save(mydir + "/losses_val",np.array(losses_val))
            torch.save(encoder, mydir + '/inter_encoder_%d' %(epoch))
            torch.save(decoder, mydir + '/inter_decoder_%d' %(epoch))
    
    # Saving model after all epochs are over
    torch.save(encoder, mydir + '/final_model_encoder')
    torch.save(decoder, mydir + '/final_model_decoder')
    
    
    
def val(epoch):
    losses_val = []
    ts = time.time()
    for iter, (images, captions, length, _) in enumerate(val_loader):
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
  
    
    
def test(sample,temp):
    encoder = torch.load('./2020-02-26_23-49-55/best_model_encoder')
    decoder1 = torch.load('./2020-02-26_23-49-55/best_model_decoder')
    for decoder_param, param in zip(decoder.parameters(), decoder1.parameters()):
        decoder_param.data.copy_(param.data)
        
    losses_test = []
    perplexities_test = []
    generated_captions = []
    meta_data_list = []
    ts = time.time()
    for iter, (images, captions, length, meta_data) in enumerate(test_loader):
        images = images.to(device)
        captions = captions.to(device)
        meta_data_list +=meta_data
        with torch.no_grad():
            image_features = encoder(images)
            generated_captions += decoder.generate(image_features,sample=sample,temperature=temp)
            output_caption = decoder(image_features, captions)
        targets= pack_padded_sequence(captions, length, batch_first=True).data
        output_caption = pack_padded_sequence(output_caption, length, batch_first=True).data        
        loss_image = criterion(output_caption, targets).item()
        losses_test.append(loss_image)
        perplexities_test.append(np.exp(loss_image))
    generate_captions(generated_captions, meta_data_list)
    l_mean = np.mean(np.array(losses_test))
    p_mean = np.mean(np.array(perplexities_test))
    BLEU1, BLEU4 = evaluate_captions('./data/annotations/captions_val2014.json','generated_captions.json')
    return l_mean, p_mean, BLEU1, BLEU4
    
if __name__ =="__main__":
#     train()
# #     #TODO : Fix test error in loss calculation
# # #     l, p = test()
# # #     print("loss mean , perplecity mean for test" , l, p)
    for i in [0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0]:
        sample = True
        print("Sampling:",sample," with temperature:",i)
        l_mean, p_mean, BLEU1, BLEU4 = test(sample,i)
        print("loss_mean:",l_mean,"perplexities_mean:",p_mean,"\n BELU1:",BLEU1," BELU 4", BLEU4)

