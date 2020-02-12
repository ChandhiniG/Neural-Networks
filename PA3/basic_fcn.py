import torch.nn as nn
from utils import *

class FCN(nn.Module):

#     def __init__(self, n_class):
#         super().__init__()
#         self.n_class = n_class
#         self.encoder = nn.Sequential(
#             self.conv1   = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, dilation=1),
#             self.bnd1    = nn.BatchNorm2d(32),
#             self.conv2   = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=1),
#             self.bnd2    = nn.BatchNorm2d(64),
#             self.conv3   = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1),
#             self.bnd3    = nn.BatchNorm2d(128),
#             self.conv4   = nn.Conv2d(128,256, kernel_size=3, stride=2, padding=1, dilation=1),
#             self.bnd4    = nn.BatchNorm2d(256),
#             self.conv5   = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, dilation=1),
#             self.bnd5    = nn.BatchNorm2d(512)
#         )
#         self.relu    = nn.ReLU(inplace=True)
#         self.decoder = nn.Sequential(
#             self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
#             self.bn1     = nn.BatchNorm2d(512),
#             self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
#             self.bn2     = nn.BatchNorm2d(256),
#             self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
#             self.bn3     = nn.BatchNorm2d(128),
#             self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
#             self.bn4     = nn.BatchNorm2d(64),
#             self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
#             self.bn5     = nn.BatchNorm2d(32)
#         )
#         self.classifier = nn.Conv2d(32,self.n_class, kernel_size=1)
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.encoder = nn.Sequential(
           nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, dilation=1),
           nn.BatchNorm2d(32),
           nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=1),
           nn.BatchNorm2d(64),
           nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1),
           nn.BatchNorm2d(128),
           nn.Conv2d(128,256, kernel_size=3, stride=2, padding=1, dilation=1),
           nn.BatchNorm2d(256),
           nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, dilation=1),
           nn.BatchNorm2d(512)
        )
        self.relu    = nn.ReLU(inplace=True)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(32)
        )
        self.classifier = nn.Conv2d(32,self.n_class, kernel_size=1)

    def forward(self, x):
        x1 = self.relu(x)
        out_encoder = self.encoder(x1)
        encoded = self.relu(out_encoder)     
        out_decoder = self.decoder(encoded)
        score = self.classifier(out_decoder)                   
        return score  # size=(N, n_class, x.H/1, x.W/1)
    
    def evaluate(self, img_batch, target_batch):
        # forward pass
        target_batch = target_batch.argmax(axis=1)
        probs_batch = self.forward(img_batch)
        pred_batch = probs_batch.argmax(axis = 1)
        p_acc = pixel_acc(pred_batch, target_batch)
        iou_acc = iou(pred_batch, target_batch,self.n_class)
        
        return p_acc, iou_acc