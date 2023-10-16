import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
from config import PARAMS

class OpticalDamageNetSkeleton(nn.Module):
    """ Define a CNN """
    def __init__(self, input_shape=(150, 150, 3), latent_dim=512):

        super(OpticalDamageNetSkeleton, self).__init__()
        self.input_shape = input_shape

        #encoder
        self.conv1 = nn.Conv2d(in_channels=input_shape[-1], out_channels=64, kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2)
        self.b_norm_1 = nn.BatchNorm2d(64)
        self.b_norm_2 = nn.BatchNorm2d(32)
        self.b_norm_3 = nn.BatchNorm2d(16)

        self.b2_norm_1 = nn.BatchNorm2d(64)
        self.b2_norm_2 = nn.BatchNorm2d(32)
        self.b2_norm_3 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()

        self.ld_channels = 16
        h, w = input_shape[:2]
        self.h, self.w = h // 2**2, w // 2**2

        self.fc1 = nn.Linear(self.h*self.w*self.ld_channels, latent_dim)
        self.fc2 = nn.Linear(latent_dim, self.h*self.w*self.ld_channels)

        #decoder
        self.dconv1 = nn.Conv2d(in_channels=self.ld_channels, out_channels=16, kernel_size=3, padding=1)
        self.dconv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=0)
        self.dconv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0)
        self.dlast = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1,  padding=0)


        self.dconv2_t = nn.ConvTranspose2d(in_channels=16, out_channels=32, kernel_size=3, output_padding=1, stride=2)
        self.dconv3_t = nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=3, output_padding=1,stride=2)
                
        
    def encode(self, x):
        x = self.pool(self.b_norm_1(self.relu(self.conv1(x))))

        x = self.pool(self.b_norm_2(self.relu(self.conv2(x))))
        
        x = self.b_norm_3(self.relu(self.conv3(x)))

        x = torch.reshape(x, (x.shape[0],-1))
        x = self.fc1(x)
        return x
    
    def decode(self,x):
        
        x = self.fc2(x)

        x = torch.reshape(x, (PARAMS.batch_size, self.ld_channels, self.h, self.w))
        
        x = self.relu(self.dconv1(self.b2_norm_3(x)))
        
        x = self.relu(self.dconv2(self.b2_norm_2(self.relu(self.dconv2_t(x)))))
        
        x = self.relu(self.dconv3(self.b2_norm_1(self.relu(self.dconv3_t(x)))))

        x = self.dlast(x)
        return x

    def forward(self, x):
        o1 = self.encode(x)
        o2 = self.decode(o1)
        return o2
    
def OpticalDamageNet(input_shape=(256,256,1)):
    return OpticalDamageNetSkeleton(input_shape)