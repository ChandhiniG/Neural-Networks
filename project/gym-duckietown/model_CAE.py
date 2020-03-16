import torch
import torchvision
import torch.nn as nn
from utils import *

class ConvAutoEncoder(nn.Module):
    '''
    This class loads a pre-trained VGG11 model to perform image segmentation on images. It:
    1. Extracts its encoding layers and freezes their weights
    2. Creates a decoder layer which is a mirror reflection of the encoding layers
    3. Defines its forward function.
    '''
    def __init__(self):
        super().__init__()
        
        # Encoder: Extracing encoder config from pretraiend model
        model_pretrained = torchvision.models.vgg11_bn(pretrained=True)
#         model_pretrained = freeze_weights(model_pretrained)
        self.encoder = list(model_pretrained.children())[0][0:15] # extracting encoding layers of VGG
        
        # Encoder: Changing maxpool layer config to change return_indices to True
        for i, layer in enumerate(self.encoder):
            if type(layer) == torch.nn.modules.pooling.MaxPool2d:
                kernel_size, stride, padding, dilation, ceil_mode = layer.kernel_size, layer.stride, layer.padding, layer.dilation, layer.ceil_mode
                layer = torch.nn.modules.pooling.MaxPool2d(kernel_size, 
                                                           stride=stride, 
                                                           padding=padding, 
                                                           dilation=dilation, 
                                                           return_indices=True, 
                                                           ceil_mode=ceil_mode)
                self.encoder[i] = layer
                
        # Decoder: Defining decoder according to encoder structure
        self.decoder = nn.Sequential(
                    nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0), #14
                    nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), #12
                    nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1), #11
                    nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), #9
                    nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1), #8
                    nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0), #7
                    nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), #5
                    nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1), #4
                    nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0), #3
                    nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), #1
                    nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1), #0
            )
        
    def forward(self, x):
        '''
        Feeds forward batch of images through the model. Visually, it can be shown as:
                image (x) --> encoder --> decoder --> output (x)
        
        :param x: input to model
        :type x: torch.Tensor
        '''
        x, indices, output_sizes = self.forward_encoder(x)
        x = self.forward_decoder(x, indices, output_sizes)
        return x
    
    def forward_encoder(self, x):
        '''
        Feeds forward batch of images through encoder
        :param x: input to model
        :type x: torch.Tensor
        '''
        indices = []
        output_sizes = []
        for i, layer in enumerate(self.encoder):
            # If layer is max pool, save the indices at which max occured and the size of the image before pooling
            if type(layer) == torch.nn.modules.pooling.MaxPool2d:
                size = x.shape[-2:]
                x, ind = layer(x)
                output_sizes.append(size)
                indices.append(ind)
            else:
                x = layer(x)
        
        return x, indices, output_sizes
    
    def forward_decoder(self, x, indices, output_sizes):
        '''
        Feeds forward batch of images through decoder
        :param x: input to model
        :type x: torch.Tensor
        :param indices: The indices at which max was detected in max pool layers
        :type: list
        :param output_sizes: The output size of image after max pool occured
        :type: list
        '''
        for i, layer in enumerate(self.decoder):
            if type(layer) == torch.nn.modules.pooling.MaxUnpool2d:
                ind = indices.pop()
                desired_size = output_sizes.pop()
                x = layer(x, ind, output_size = desired_size)
            else:
                x = layer(x)
        
        assert len(indices) == len(output_sizes) == 0, 'Imbalance in number of max pool and unpool layers'
        return x