from torchvision import utils
from basic_fcn import *
from dataloader import *
from utils import *
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time

transforms_composed = transforms.Compose([
                        transforms.Resize((512, 256)),
#                         transforms.RandomRotation(degrees=30),
#                         transforms.RandomVerticalFlip(p=0.5),
])

# Apply transformation if needed
apply_transform = False

if apply_transform:
    train_dataset = CityScapesDataset(csv_file='train_small.csv', transforms = transforms_composed)
else:
    train_dataset = CityScapesDataset(csv_file='train_small.csv')
val_dataset = CityScapesDataset(csv_file='val_small.csv')
test_dataset = CityScapesDataset(csv_file='test_small.csv')
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


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform(m.weight.data)
        torch.nn.init.zeros_(m.bias.data)
        
epochs     = 50
criterion = nn.CrossEntropyLoss()
fcn_model = FCN(n_class=n_class)
###changing class here
fcn_model = FCN_segnet(n_class=n_class)
fcn_model.apply(init_weights)
#fcn_model = torch.load('best_model')
optimizer = optim.Adam(fcn_model.parameters(), lr=5e-3)


use_gpu = torch.cuda.is_available()
if use_gpu:
    fcn_model = fcn_model.cuda()
    
def train():
    losses = []
    losses_val = []
    p_accs = []
    iou_accs = []
    fcn_model.train()
    min_loss = 100
    for epoch in range(epochs+1):
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
        fcn_model.train()
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

def val(epoch,flag = True):
    # Complete this function - Calculate loss, accuracy and IoU for every epoch
    # Make sure to include a softmax after the output from your model
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
#         mask = np.logical_not(np.isnan(iou))
#         iou_acc += np.mean(iou[mask])
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
#    val(0)# show the accuracy before training
#     test()
    train()
    print("Testing ")
    test()







