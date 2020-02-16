import torch.nn as nn
from utils import *

class FCN(nn.Module):
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
    
    def evaluate(self, img_batch, target_batch,target_y):
        # forward pass
        with torch.no_grad():
            probs_batch = self.forward(img_batch)
        target_batch = target_batch.argmax(dim=1)
        pred_batch = probs_batch.argmax(dim = 1)
#         p_acc = pixel_acc(pred_batch, target_batch)
        p_acc2 = pixel_acc2(pred_batch, target_y)
        iou2_ints,iou2_unions = iou2(pred_batch,target_y,self.n_class)
        
        return p_acc2, iou2_ints, iou2_unions


class FCN_vgg(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.maxpool = nn.MaxPool2d(2, return_indices=True)
        self.unmaxpool = nn.MaxUnpool2d(2)
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(3,8, kernel_size=5, stride=2, padding=2, dilation=1),
            nn.BatchNorm2d(8),
            nn.Conv2d(8,16, kernel_size=5, stride=2, padding=2, dilation=1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16,32, kernel_size=5, stride=2, padding=2, dilation=1),
            nn.BatchNorm2d(32))

        self.encoder_2 = nn.Sequential(
            nn.Conv2d(32,64, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,128, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(128),
            nn.Conv2d(128,256, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(256))
        
        self.encoder_3 = nn.Sequential(
            nn.Conv2d(256,512, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(512))

        self.relu    = nn.PReLU()

        self.decoder_1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1, dilation=1, output_padding=0),
            nn.BatchNorm2d(256))
        
        self.decoder_2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1, dilation=1, output_padding=0),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1, dilation=1, output_padding=0),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, dilation=1, output_padding=0),
            nn.BatchNorm2d(32))

        self.decoder_3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=2, dilation=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 8, kernel_size=5, stride=2, padding=2, dilation=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ConvTranspose2d(8, 3, kernel_size=5, stride=2, padding=2, dilation=1, output_padding=1),
            nn.BatchNorm2d(3))

        self.classifier = nn.Conv2d(3,self.n_class, kernel_size=1)

    def forward(self, x):
        x1 = self.relu(x)
        ##encoder 
        out_1 = self.encoder_1(x1)
        out_m_1, indices_1 = self.maxpool(out_1)
        
        out_2 = self.encoder_2(out_m_1)
        out_m_2, indices_2 = self.maxpool(out_2)

        out_3 = self.encoder_3(out_m_2)
        out_m_3, indices_3 = self.maxpool(out_3)

        encoded = self.relu(out_m_3)  

        ###decoder 
        out_d_1 = self.unmaxpool(encoded, indices_3, output_size=out_3.size())
        out_decoder_1 = self.decoder_1(out_d_1)

        out_d_2 = self.unmaxpool(out_decoder_1, indices_2, output_size=out_2.size())
        out_decoder_2 = self.decoder_2(out_d_2)
        
        out_d_3 = self.unmaxpool(out_decoder_2, indices_1, output_size=out_1.size())
        out_decoder_3 = self.decoder_3(out_d_3)

        score = self.classifier(out_decoder_3)                   
        return score  # size=(N, n_class, x.H/1, x.W/1)
    
    def evaluate(self, img_batch, target_batch,target_y):
        # forward pass
        with torch.no_grad():
            probs_batch = self.forward(img_batch)
        target_batch = target_batch.argmax(dim=1)
        pred_batch = probs_batch.argmax(dim = 1)
#         p_acc = pixel_acc(pred_batch, target_batch)
        p_acc2 = pixel_acc2(pred_batch, target_y)
        iou2_ints,iou2_unions = iou2(pred_batch,target_y,self.n_class)
        return p_acc2, iou2_ints, iou2_unions
    
   
    
# In val
# torch.Size([2, 3, 1024, 2048]) torch.Size([2, 34, 1024, 2048])
# In evaluate
# torch.Size([2, 3, 1024, 2048]) torch.Size([2, 34, 1024, 2048])
# pred and target batch
# torch.Size([2, 1024, 2048]) torch.Size([2, 1024, 2048])
# In pixel accuracy
# torch.Size([2, 1024, 2048]) torch.Size([2, 1024, 2048])
# torch.Size([])
