# Training script for UNet

from torchvision import utils
from unet_fcn import *
from dataloader import *
from utils import *
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time

batch_s = 2
num_w = 10
apply_transform = True

transforms_composed = transforms.Compose([
                        transforms.Resize((1024, 512)),
#                         transforms.RandomRotation(degrees=30),
#                         transforms.RandomVerticalFlip(p=0.5),
])

# Apply transformation if needed
if apply_transform:
    train_dataset = CityScapesDataset(csv_file='train_small.csv', transforms = transforms_composed)
else:
    train_dataset = CityScapesDataset(csv_file='train_small.csv')
val_dataset = CityScapesDataset(csv_file='val_small.csv')
test_dataset = CityScapesDataset(csv_file='test_small.csv')
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_s,
                          num_workers=num_w,
                          shuffle=True)
val_loader = DataLoader(dataset=val_dataset,
                          batch_size=batch_s,
                          num_workers=num_w,
                          shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                          batch_size=batch_s,
                          num_workers=num_w,
                          shuffle=True)

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform(m.weight.data)
        torch.nn.init.zeros_(m.bias.data)
        
epochs     = 100
criterion = nn.CrossEntropyLoss()
fcn_model = UNet(n_class=34)
fcn_model.apply(init_weights)
#fcn_model = torch.load('best_model')
optimizer = optim.Adam(fcn_model.parameters(), lr=5e-3)

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Access to GPU, you did it")
    fcn_model = fcn_model.cuda()
    
def train():
    losses = []
    p_accs = []
    iou_accs = []
    fcn_model.train()
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
                print("Training --- epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))
                break

        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        losses.append(np.mean(np.array(losses_epoch)))
        p_acc,iou_acc = val(epoch)
        p_accs.append(p_acc.item())
        iou_accs.append(iou_acc)
        fcn_model.train()
        if epoch%10 == 0:
            torch.save(fcn_model, 'unet_best_model')
            np.save("unet_losses",np.array(losses))
            np.save("unet_p_acc",np.array(p_accs))
            np.save("unet_iou_acc",np.array(iou_accs))


def val(epoch):
    # Complete this function - Calculate loss, accuracy and IoU for every epoch
    # Make sure to include a softmax after the output from your model
    p_acc = 0
    iou_acc = 0
    count = 0
    fcn_model.eval()
    for iter, (X, tar, Y) in enumerate(val_loader):
        if use_gpu:
            X = X.cuda()
            tar = tar.cuda()
        else:
            X, tar = X.cpu(), tar.cpu()
        p, iou = fcn_model.evaluate(X, tar)
        p_acc += p
        iou = np.array(iou)
        mask = np.logical_not(np.isnan(iou))
        iou_acc += np.mean(iou[mask])
        count += 1
    print("Epoch {}: Pixel Acc: {}, IOU Acc: {}".format(epoch, p_acc / count, iou_acc / count))
    return p_acc / count, iou_acc / count


def test():
    # Complete this function - Calculate accuracy and IoU
    # Make sure to include a softmax after the output from your model
    p_acc = 0
    iou_acc = 0
    count = 0
    fcn_model.eval()
    for iter, (X, tar, Y) in enumerate(test_loader):
        if use_gpu:
            X = X.cuda()
            tar = tar.cuda()
        else:
            X, tar = X.cpu(), tar.cpu()
        p, iou = fcn_model.evaluate(X, tar)
        p_acc += p
        iou = np.array(iou)
        mask = np.logical_not(np.isnan(iou))
        iou_acc += np.mean(iou[mask])
        count += 1
    print("Pixel Acc: {}, IOU Acc: {}".format(p_acc / count, iou_acc / count))
    return p_acc / count, iou_acc / count


if __name__ == "__main__":
    #    val(0)# show the accuracy before training
    #     test()
    train()
    print("Testing ")
    test()
