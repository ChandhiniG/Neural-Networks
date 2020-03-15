import torch
import torch.nn as nn

class ConvAutoEncoder(nn.Module):
    '''
    This class loads a pre-trained VGG11 model to perform image segmentation on images. It:
    1. Extracts its encoding layers and freezes their weights
    2. Creates a decoder layer which is a mirror reflection of the encoding layers
    3. Defines its forward function.
    '''
    def __init__(self):
        super().__init__()
        
        # Encoder: (Important: Set return_indices = True for MaxPool2d layers)
        self.encoder = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=True, ceil_mode=False),
                    nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=True, ceil_mode=False),
                    nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=True, ceil_mode=False),
            )
                
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