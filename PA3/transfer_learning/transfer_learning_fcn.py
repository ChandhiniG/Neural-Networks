import torchvision
import torch.nn as nn
from utils import *

class PretrainedEncoder(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        
        # Encoder: Extracing encoder config from pretraiend model
        model_pretrained = torchvision.models.vgg11(pretrained=True)
        self.encoder = list(model_pretrained.children())[0] #type nn.Sequential
        
        # Encoder: Changing maxpool layer config to change return_indices to True
        for i, layer in enumerate(self.encoder):
            if type(layer) == torch.nn.modules.pooling.MaxPool2d:
                kernel_size, stride, padding, dilation, ceil_mode = layer.kernel_size, \
                                                                    layer.stride, layer.padding, layer.dilation, layer.ceil_mode
                layer = torch.nn.modules.pooling.MaxPool2d(kernel_size, 
                                                           stride=stride, 
                                                           padding=padding, 
                                                           dilation=dilation, 
                                                           return_indices=True, 
                                                           ceil_mode=ceil_mode)
                self.encoder[i] = layer
                
        # Decoder: Defining decoder according to encoder
        self.decoder = nn.Sequential(
                    nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0), #20
                    nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1), #18
                    nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1), #16
                    nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0), #15
                    nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1), #13
                    nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1), #11
                    nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0), #10
                    nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1), #8
                    nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1), #6
                    nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0), #5
                    nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1), #3
                    nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0), #2            
                    nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1), #1
            )
        
        self.classifier = nn.Conv2d(3,self.n_class, kernel_size=1)
        
    def forward(self, x):
        indices = []
        output_sizes = []
        # Encoder: Forward pass
        for i, layer in enumerate(self.encoder):
            # If layer is max pool, save the indices at which max occured and the size of the image before pooling
            if type(layer) == torch.nn.modules.pooling.MaxPool2d:
                size = x.shape[-2:]
                x, ind = layer(x)
                output_sizes.append(size)
                indices.append(ind)
            else:
                x = layer(x)
        
        # Decoder: Forward pass
        for i, layer in enumerate(self.decoder):
            if type(layer) == torch.nn.modules.pooling.MaxUnpool2d:
                ind = indices.pop()
                desired_size = output_sizes.pop()
                x = layer(x, ind, output_size = desired_size)
            else:
                x = layer(x)
        assert len(indices) == len(output_sizes) == 0, 'Imbalance in number of max pool and unpool 2d'
        
        # Classifier: Going from 3 to n_class channels
        x = self.classifier(x)
        
        return x
    
    def evaluate(self, img_batch, target_batch,target_y):
        with torch.no_grad():
            probs_batch = self.forward(img_batch)
        pred_batch = probs_batch.argmax(dim = 1)
#         p_acc = pixel_acc(pred_batch, target_batch)
        p_acc2 = pixel_acc2(pred_batch, target_y)
        iou2_ints,iou2_unions = iou2(pred_batch,target_y,self.n_class)
        
        return p_acc2, iou2_ints, iou2_unions