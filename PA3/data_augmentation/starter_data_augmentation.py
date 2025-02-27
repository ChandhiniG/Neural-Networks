import sys
sys.path.insert(0,'..') #adding parent directory to module search path

from torchvision import utils
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time
import argparse

from basic_fcn import *
from dataloader_data_augmentation import *
from utils import *

def init_weights(m):
    '''
    Initializing weight of Decoder layers only
    '''
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.zeros_(m.bias.data)

# ------------- Please read. Thanks. You're awesome -------------
# NOTE: Data augmentation is hard wired in the dataset class of dataloader_data_augmentation.py file. The transformations
# had to be done manually so that the SAME random transformation is applied to the img and label.
# Solution inspired from - https://discuss.pytorch.org/t/how-to-apply-same-transform-on-a-pair-of-picture/14914/2
# ---------------------------------------------------------------

train_dataset = CityScapesDataset(csv_file='../train.csv')
val_dataset = CityScapesDataset(csv_file='../val.csv')
test_dataset = CityScapesDataset(csv_file='../test.csv')
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
fcn_model = FCN(n_class=n_class)
fcn_model.apply(init_weights)
optimizer = optim.Adam(fcn_model.parameters(), lr=5e-3)

use_gpu = torch.cuda.is_available()
if use_gpu:
    fcn_model = fcn_model.cuda()
    
def train():
    '''
    This function trains the model, stores the traing and validation loss, 
    stores the best model model every epoch, stores the losses every 5 epochs,
    and saves the model at the end of the training irrespective of whether it was best or not
    '''
    losses = []
    losses_val = []
    p_accs = []
    iou_accs = []
    min_loss = float('inf')
    for epoch in range(epochs+1):
        fcn_model.train()
        losses_epoch = []
        ts = time.time()
        for iter, (X, tar, Y) in enumerate(train_loader):
            optimizer.zero_grad()

            if use_gpu:
                inputs = X.cuda()
                labels = Y.cuda()
            else:
                inputs, labels = X.cpu(), Y.cpu()

            outputs = fcn_model(inputs)
            loss = criterion(outputs, labels)
            losses_epoch.append(loss.item())
            loss.backward()
            optimizer.step()
            
            if iter % 400 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))
        
        losses.append(np.mean(np.array(losses_epoch)))
        print("Finish epoch {}, time elapsed {:.2f}, loss {:.3f}".format(epoch, time.time() - ts, losses[-1]))
        losses_val.append(val(epoch))
        if(min_loss>losses_val[-1]):
            torch.save(fcn_model, 'best_model')
            min_loss = losses_val[-1]
        if epoch%5 == 0:
            np.save("losses",np.array(losses))
            np.save("losses_val",np.array(losses_val))

    torch.save(fcn_model, 'final_model')
    p_acc,iou_acc = val(epochs,False)
    np.save("p_acc",np.array([p_acc]))
    np.save("iou_acc",np.array([iou_acc]))
    print("pixel accuracy", p_acc , "iou acc", iou_acc)

def val(epoch,flag = True):
    '''
    This function tests the model using the validation set.
    
    The input epoch tells it at which epoch its performing validation.
    If flag = True, it will only return validation loss
    If flag = False, it will return pixel accuracy and iou accuracy
    '''
    p_acc = 0
    iou_acc = 0
    iou_int = []
    iou_union = []
    count = 0
    fcn_model.eval()
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
                outputs = fcn_model(X)
            losses.append(criterion(outputs, Y).item())
            continue
        p, iou_i, iou_u = fcn_model.evaluate(X, tar,Y)
        p_acc += p
        iou_int.append(iou_i) 
        iou_union.append(iou_u) 
        count += 1
    if flag:
        return np.mean(np.array(losses))
    iou_int = np.sum(np.array(iou_int),axis=0)
    iou_union = np.sum(np.array(iou_union),axis=0)
    iou_union += 1e-10 #to avoid zero division error
    iou_acc = np.mean(iou_int/iou_union)
    print("Epoch {}: Pixel Acc: {}, IOU Acc: {}".format(epoch, p_acc/count, iou_acc))
    return p_acc/count, iou_acc

def test():
    '''
    Function to test the model and output the pixel and iou accuracy
    '''
    p_acc = 0
    iou_acc = 0
    iou_int = []
    iou_union = []
    count = 0
    for iter, (X, tar, Y) in enumerate(test_loader):
        if use_gpu:
            X = X.cuda()
            tar = tar.cuda()
            Y = Y.cuda()
        else:
            X,tar,Y = X.cpu(), tar.cpu(),Y.cpu()
        p, iou_i, iou_u = fcn_model.evaluate(X, tar,Y)
        p_acc += p
        iou_int.append(iou_i) 
        iou_union.append(iou_u) 
        count += 1
    iou_int = np.sum(np.array(iou_int),axis=0)
    iou_union = np.sum(np.array(iou_union),axis=0)
    iou_union += 1e-10 #to avoid zero division error
    iou_acc = np.mean(iou_int/iou_union)
    print("Test : Pixel Acc: {}, IOU Acc: {}".format( p_acc/count, iou_acc))
    return p_acc/count, iou_acc
      
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', required = True)
    args = parser.parse_args()
    
    if args.mode == "train":
        train()
    else:
        print("Testing ")
        model = torch.load('best_model')
        pixel_accuracy, iou_accuracy = test()
        print('pixel_accuracy = ', pixel_accuracy)
        print('iou_accuracy = ', iou_accuracy)
