import torch.nn as nn
import torch
from utils import *
import copy

class UNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.contract_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1, dilation=1),
                nn.ReLU(inplace=True)),
            nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1, dilation=1),
                nn.ReLU(inplace=True)),
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1, dilation=1),
                nn.ReLU(inplace=True)),
            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1, dilation=1),
                nn.ReLU(inplace=True)),
            nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1, dilation=1),
                nn.ReLU(inplace=True))])

        self.upconv_layers = nn.ModuleList([
            nn.ConvTranspose2d(1024, 512, kernel_size=3, padding=1, stride=2, dilation=1, output_padding=1),
            nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1, stride=2, dilation=1, output_padding=1),
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1, stride=2, dilation=1, output_padding=1),
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1, stride=2, dilation=1, output_padding=1),
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1, stride=2, dilation=1, output_padding=1)])

        self.upconv_layers2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1024, 512, kernel_size=3, padding=1, stride=1, dilation=1),
                nn.ReLU(inplace=True)),
            nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1, dilation=1),
                nn.ReLU(inplace=True)),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=3, padding=1, stride=1, dilation=1),
                nn.ReLU(inplace=True)),
            nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1, dilation=1),
                nn.ReLU(inplace=True)),
            ])
        
        self.bottom = nn.Sequential(
                nn.Conv2d(512, 1024, kernel_size=3, padding=1, stride=1, dilation=1),
                nn.ReLU(inplace=True))
        self.maxpool = nn.MaxPool2d(2)
        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)

    def forward(self, x):
        contracted = []
        for layer in self.contract_layers:
            x = layer(x)
#             print(f"contract- {x.shape}")
            contracted.append(copy.copy(x))
            x = self.maxpool(x)
#             print(f"maxpooled- {x.shape}")
            
        x = self.bottom(x)
#         print(f"bottom- {x.shape}")
        x = self.upconv_layers[0](x)
#         print(f"upconv1- {x.shape}")
        x = torch.cat((contracted[4], x), dim=1)
#         print(f"concat-- {x.shape}")
        x = self.upconv_layers2[0](x)
#         print(f"upconv2--- {x.shape}")
        
#         print(f"bottom- {x.shape}")
        x = self.upconv_layers[1](x)
#         print(f"upconv1- {x.shape}")
        x = torch.cat((contracted[3], x), dim=1)
#         print(f"concat-- {x.shape}")
        x = self.upconv_layers2[1](x)
#         print(f"upconv2--- {x.shape}")

        x = self.upconv_layers[2](x)
        x = torch.cat((contracted[2], x), dim=1)
        x = self.upconv_layers2[2](x)
#         print(f"upconv2--- {x.shape}")

        x = self.upconv_layers[3](x)
        x = torch.cat((contracted[1], x), dim=1)
        x = self.upconv_layers2[3](x)
#         print(f"upconv2--- {x.shape}")
        
        x = self.upconv_layers[4](x)
        score = self.classifier(x)
#         print(f"upconv2--- {score.shape}")
        return score  # size=(N, n_class, x.H/1, x.W/1)

    def evaluate(self, img_batch, target_batch):
        # forward pass
        target_batch = target_batch.argmax(dim=1)
        probs_batch = self.forward(img_batch)
        pred_batch = probs_batch.argmax(dim=1)
        p_acc = pixel_acc(pred_batch, target_batch)
        iou_acc = iou(pred_batch, target_batch, self.n_class)

        return p_acc, iou_acc
