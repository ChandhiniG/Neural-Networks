import torch
import torch.nn as nn

class ConvAutoEncoder(nn.Module):
    '''
    This class creates a VGG11 inpsired auto encoder model. It:
    1. Creates encoder layers
    2. Creates decoder layer,which is a mirror reflection of the encoding layers
    3. Defines two forward function - one for the encoder and decoder each
    '''
    def __init__(self):
        super().__init__()
        
        # Encoder: (Important: Set return_indices = True for MaxPool2d layers)
        self.encoder = nn.Sequential(
                    nn.Conv2d(3, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=True, ceil_mode=False),
                    
                    nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=True, ceil_mode=False),
                    
                    nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=True, ceil_mode=False),
            
                    nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=True, ceil_mode=False),
                    
                    nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=True, ceil_mode=False),
            )
                
        # Decoder: Defining decoder according to encoder structure
        self.decoder = nn.Sequential(
                    nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0),
                    nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1),
                    
                    nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0),
                    nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
                    nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1),
            
                    nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0),
                    nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ConvTranspose2d(16, 16, kernel_size=3, stride=1, padding=1),
            
                    nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0),
                    nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ConvTranspose2d(16, 8, kernel_size=3, stride=1, padding=1),
                    
                    nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0),
                    nn.BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ConvTranspose2d(8, 3, kernel_size=3, stride=1, padding=1),
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