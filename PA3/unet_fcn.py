# The U-Net model

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
                nn.ReLU(inplace=True))])

        self.upconv_layers = nn.ModuleList([
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1, stride=2, dilation=1, output_padding=1),
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1, stride=2, dilation=1, output_padding=1),
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1, stride=2, dilation=1, output_padding=1)])

        self.upconv_layers2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=3, padding=1, stride=1, dilation=1),
                nn.ReLU(inplace=True)),
            nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1, dilation=1),
                nn.ReLU(inplace=True))
        ])

        self.bottom = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1, dilation=1),
            nn.ReLU(inplace=True))
        self.maxpool = nn.MaxPool2d(2)
        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)

    def forward(self, x):

        # the downwards part of the U
        contracted = []
        for layer in self.contract_layers:
            x = layer(x)
            contracted.append(copy.copy(x))
            x = self.maxpool(x)

        # the upwards part of the U
        x = self.bottom(x)
        x = self.upconv_layers[0](x)
        x = torch.cat((contracted[2], x), dim=1)
        x = self.upconv_layers2[0](x)
        x = self.upconv_layers[1](x)
        x = torch.cat((contracted[1], x), dim=1)
        x = self.upconv_layers2[1](x)
        x = self.upconv_layers[2](x)

        score = self.classifier(x)

        return score  # size=(N, n_class, x.H/1, x.W/1)

    def evaluate(self, img_batch, target_batch, target_y):
        # forward pass
        with torch.no_grad():
            probs_batch = self.forward(img_batch)
        pred_batch = probs_batch.argmax(dim=1)
        p_acc2 = pixel_acc2(pred_batch, target_y)
        iou2_ints, iou2_unions = iou2(pred_batch, target_y, self.n_class)

        return p_acc2, iou2_ints, iou2_unions
