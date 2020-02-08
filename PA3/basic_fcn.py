import torch.nn as nn
from utils import *

class FCN(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.conv1   = nn.Conv2d(__, 32, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd1    = nn.BatchNorm2d(__)
        self.conv2   = nn.Conv2d(__, 64, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd2    = nn.BatchNorm2d(__)
        self.conv3   = nn.Conv2d(__, 128, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd3    = nn.BatchNorm2d(__)
        self.conv4   = nn.Conv2d(__,256, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd4    = nn.BatchNorm2d(__)
        self.conv5   = nn.Conv2d(__, 512, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd5    = nn.BatchNorm2d(__)
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(__, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(__)
        self.deconv2 = nn.ConvTranspose2d(__, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(__)
        self.deconv3 = nn.ConvTranspose2d(__, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(__)
        self.deconv4 = nn.ConvTranspose2d(__, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(__)
        self.deconv5 = nn.ConvTranspose2d(__, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(__)
        self.classifier = nn.Conv2d(__, __, kernel_size=1)

    def forward(self, x):
        x1 = __(self.relu(__(x)))
        # Complete the forward function for the rest of the encoder

        score = __(self.relu(__(out_encoder)))     
        # Complete the forward function for the rest of the decoder
        
        score = self.classifier(out_decoder)                   

        return score  # size=(N, n_class, x.H/1, x.W/1)
    
    def eval(self, img_batch, target_batch):
        # forward pass
        target_batch = target_batch.argmax(axis=1)
        probs_batch = self.forward(img_batch)
        pred_batch = probs_batch.argmax(axis = 1)
        pixel_acc = pixel_acc(pred_batch, target_batch)
        iou_acc = iou(pred_batch, target_batch)
        
        return pixel_acc, iou_acc