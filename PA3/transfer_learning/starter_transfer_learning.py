import sys
sys.path.insert(0,'..') #adding parent directory to module search path

import torch
import torchvision
from torchvision import utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time
# from tensorboardX import SummaryWriter

from dataloader import *
from utils import *
from transfer_learning_fcn import *

def init_weights(m):
    '''
    Initializing weight of Decoder layers only
    '''
    if isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.zeros_(m.bias.data)

def freeze_encoder_weights(model):
    '''
    Freezes the weights for the encoder layer. This is based on the fact that the encoder is a nn.Sequential object
    and is named encoder. Check the model in transfer_learning_fcn.py file for clarity
    '''
    for name, param in model.named_parameters():
        if 'encoder' in name:
            param.requires_grad = False
    return model

transforms_composed = transforms.Compose([
                        transforms.Resize((256, 512)),
                        ])
train_dataset = CityScapesDataset(csv_file='../train.csv', transforms = transforms_composed)
val_dataset = CityScapesDataset(csv_file='../val.csv', transforms = transforms_composed)
test_dataset = CityScapesDataset(csv_file='../test.csv', transforms = transforms_composed)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=2,
                          num_workers=10,
                          shuffle=True)
val_loader = DataLoader(dataset=val_dataset,
                          batch_size=2,
                          num_workers=10,
                          shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                          batch_size=2,
                          num_workers=10,
                          shuffle=True)

epochs    = 60
criterion = nn.CrossEntropyLoss()
model     = PretrainedEncoder(n_class=n_class)
model     = freeze_encoder_weights(model)
model.apply(init_weights)
optimizer = optim.Adam(model.parameters(), lr=5e-3)

use_gpu = torch.cuda.is_available()
if use_gpu:
    model = model.cuda()
    
def train():
    losses, losses_val = [], []
    p_accs, iou_accs = [], []
    min_loss = float('inf')
#     anchor_X, _, anchor_Y = train_dataset[206] #for tensorboardX
#     anchor_X = anchor_X.unsqueeze_(0) #adding an extra dimension to fake batchsize of 1
    for epoch in range(epochs+1):
        model.train()
        losses_epoch = []
        ts = time.time()
        ###  Start: Training with mini batches over the dataset
        for i, (X, tar, Y) in enumerate(train_loader):
            optimizer.zero_grad()

            if use_gpu:
                inputs = X.cuda()
                labels = Y.cuda()
            else:
                inputs, labels = X.cpu(), Y.cpu()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses_epoch.append(loss.item())
            loss.backward()
            optimizer.step()
            
            if i % 400 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, i, loss.item()))
        ###  End: Training with mini batches over the dataset
        
        losses.append(np.mean(np.array(losses_epoch)))
        print("Finish epoch {}, time elapsed {:.2f}, loss: {:.3f}".format(epoch, time.time() - ts, losses[-1]))
        losses_val.append(val(epoch))
        
#         ### Start: TensorboardX extra code
#         writer.add_scalar('data/train_loss', losses[-1], i)
#         writer.add_scalar('data/val_loss', losses_val[-1], i)
#         ## End: TensorboardX extra code
        
        # Saving model if its the best
        if(min_loss>losses_val[-1]):
            torch.save(model, 'best_model')
            min_loss = losses_val[-1]
        # Saving train and val loss every 5 epochs
        if epoch%5 == 0:
            np.save("losses",np.array(losses))
            np.save("losses_val",np.array(losses_val))
#             ### Start: TensorboardX extra code
#             pred = model(anchor_X.cuda())
#             pred_Y = pred.argmax(dim = 1)
#             pred_Y = pred_Y.cpu().data.numpy()
#             # Convert to 1D image
#             writer.add_image('image/actual_Y', anchor_Y, i)
#             writer.add_image('image/pred_Y', pred_Y[0], i)
#             ### End: TensorboardX extra code

    torch.save(model, 'final_model')
    p_acc,iou_acc = val(epochs,False)
    np.save("p_acc",np.array([p_acc]))
    np.save("iou_acc",np.array([iou_acc]))
    print("pixel accuracy", p_acc , "\niou acc", iou_acc)

def val(epoch,flag = True):
    '''
    If flag = True, it will only return validation loss
    If flag = False, it will return pixel accuracy and iou accuracy
    '''
    p_acc, iou_acc, count = 0, 0, 0
    iou_int, iou_union = [], []
    
    model.eval()
    losses = []
    for iter, (X, tar, Y) in enumerate(val_loader):
        if use_gpu:
            X = X.cuda()
            tar = tar.cuda()
            Y = Y.cuda()
        else:
            X,tar,Y = X.cpu(), tar.cpu(),Y.cpu()
            
        if flag:
            with torch.no_grad():
                outputs = model(X)
            losses.append(criterion(outputs, Y).item())
            continue
        p, iou_i, iou_u = model.evaluate(X, tar,Y)
        p_acc += p
        iou_int.append(iou_i) 
        iou_union.append(iou_u) 
        count += 1

    if flag:
        return np.mean(np.array(losses))
    
    iou_int = np.sum(np.array(iou_int),axis=0)
    iou_union = np.sum(np.array(iou_union),axis=0)
    iou_acc = np.mean(iou_int/iou_union)
    print("Epoch {}: Pixel Acc: {}, IOU Acc: {}".format(epoch, p_acc/count, iou_acc))
    
    return p_acc/count, iou_acc
    
    
def test():
    #Complete this function - Calculate accuracy and IoU 
    # Make sure to include a softmax after the output from your model
    p_acc, iou_acc, count = 0, 0, 0
    iou_int, iou_union = [], []
    model.eval()
    
    for iter, (X, tar, Y) in enumerate(test_loader):
        if use_gpu:
            X = X.cuda()
            tar = tar.cuda()
            Y = Y.cuda()
        else:
            X,tar,Y = X.cpu(), tar.cpu(),Y.cpu()
        p, iou_i, iou_u = model.evaluate(X, tar,Y)
        p_acc += p
        iou_int.append(iou_i) 
        iou_union.append(iou_u) 
        count += 1
    
    iou_int = np.sum(np.array(iou_int),axis=0)
    iou_union = np.sum(np.array(iou_union),axis=0)
    iou_union += 1e-10
    iou_acc = np.mean(iou_int/iou_union)
    print("Test : Pixel Acc: {}, IOU Acc: {}".format( p_acc/count, iou_acc))
    
    return p_acc/count, iou_acc
    
    
if __name__ == "__main__":
#    val(0)# show the accuracy before training
#     test()
#     writer = SummaryWriter('./tensorboard_transfer_learning')
    train()
    print("Testing ")
    pixel_accuracy, iou_accuracy = test()
    print('pixel_accuracy = ', pixel_accuracy)
    print('iou_accuracy = ', iou_accuracy)
#     writer.close()
