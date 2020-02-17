from torchvision import utils
from basic_fcn import *
from dataloader import *
from utils import *
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time
import argparse

# Apply transformation if needed, only to the train dataset
transforms_composed = transforms.Compose([transforms.Resize((512,1024))])
apply_transform = False
if apply_transform:
    train_dataset = CityScapesDataset(csv_file='train.csv', transforms = transforms_composed)
else:
    train_dataset = CityScapesDataset(csv_file='train.csv')
    
# Load val and test datasets
val_dataset = CityScapesDataset(csv_file='val.csv')
test_dataset = CityScapesDataset(csv_file='test.csv')

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
                          num_workers=3,
                          shuffle=True)

# Xavier Initialisation
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform(m.weight.data)
        torch.nn.init.zeros_(m.bias.data)
        
# Setting parameters and creating the model        
epochs     = 100
criterion = nn.CrossEntropyLoss()
fcn_model = FCN(n_class=n_class)
fcn_model.apply(init_weights)
optimizer = optim.Adam(fcn_model.parameters(), lr=5e-3)
use_gpu = torch.cuda.is_available()
if use_gpu:
    fcn_model = fcn_model.cuda()
    
"""
Trains the model and saves the best model based on the minimum validation loss
seen during training. 
"""
def train():
    losses = []
    losses_val = []
    p_accs = []
    iou_accs = []
    min_loss = 100
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
            
            if iter % 100 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))
        
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        losses.append(np.mean(np.array(losses_epoch)))
        losses_val.append(val(epoch))
        print(losses[-1],losses_val[-1])
        if(min_loss>losses_val[-1]):
            torch.save(fcn_model, 'best_model')
            min_loss = losses_val[-1]
        if epoch%10 == 0:
            np.save("losses",np.array(losses))
            np.save("losses_val",np.array(losses_val))

    torch.save(fcn_model, 'final_model')
    p_acc,iou_acc = val(epochs,False)
    np.save("p_acc",np.array([p_acc]))
    np.save("iou_acc",np.array([iou_acc]))
    print("pixel accuracy", p_acc , "iou acc", iou_acc)

"""
Calculates the pixel accuracy and iou accuracy per class for the validation dataset
"""
def val(epoch,flag = True):
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
    iou_acc = np.mean(iou_int/iou_union)
    print("Epoch {}: Pixel Acc: {}, IOU Acc: {}".format(epoch, p_acc/count, iou_acc))
    print("building{}, traffic sign{}, person{}, car{}, bicycle{}".format(
        iou_acc[2],iou_acc[7],iou_acc[11],iou_acc[13],iou_acc[18]))
    return p_acc/count, iou_acc

"""
Calculates the pixel accuracy and iou accuracy per class for the Test dataset
"""
def test():
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
        fcn_model = torch.load('best_model')
        test()

