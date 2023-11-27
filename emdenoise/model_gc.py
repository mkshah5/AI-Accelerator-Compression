import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np

class EMDenoiseNetSkeleton(nn.Module):
    """ Define a CNN """
    def __init__(self, input_shape=(256,256,1)):

        super(EMDenoiseNetSkeleton, self).__init__()
        self.input_shape = input_shape

        #encoder
        self.block1 = []
        self.block1.append(nn.Conv2d(self.input_shape[-1], 8, kernel_size=3, padding=1))
        self.block1.append(nn.ReLU())
        self.block1.append(nn.BatchNorm2d(8))
        self.block1.append(nn.Conv2d(8, 8, kernel_size=3, padding=1))
        self.block1.append(nn.ReLU())
        self.block1.append(nn.BatchNorm2d(8))
        self.b1_end = nn.MaxPool2d(2)
        self.b1 = nn.Sequential(*self.block1)

        self.block2 = []
        self.block2.append(nn.Conv2d(8, 16, kernel_size=3, padding=1))
        self.block2.append(nn.ReLU())
        self.block2.append(nn.BatchNorm2d(16))
        self.block2.append(nn.Conv2d(16, 16, kernel_size=3, padding=1))
        self.block2.append(nn.ReLU())
        self.block2.append(nn.BatchNorm2d(16))
        # self.block2.append(nn.MaxPool2d(2))
        self.b2_end = nn.MaxPool2d(2)
        self.b2 = nn.Sequential(*self.block2)

        self.block3 = []
        self.block3.append(nn.Conv2d(16, 32, kernel_size=3, padding=1))
        self.block3.append(nn.ReLU())
        self.block3.append(nn.BatchNorm2d(32))
        self.block3.append(nn.Conv2d(32, 32, kernel_size=3, padding=1))
        self.block3.append(nn.ReLU())
        self.block3.append(nn.BatchNorm2d(32))
        # self.block3.append(nn.MaxPool2d(2))
        self.b3_end = nn.MaxPool2d(2)
        self.b3 = nn.Sequential(*self.block3)

        self.block4 = []
        self.block4.append(nn.Conv2d(32, 64, kernel_size=3, padding=1))
        self.block4.append(nn.ReLU())
        self.block4.append(nn.BatchNorm2d(64))
        self.block4.append(nn.Conv2d(64, 64, kernel_size=3, padding=1))
        self.block4.append(nn.ReLU())
        self.block4.append(nn.BatchNorm2d(64))
        self.b4 = nn.Sequential(*self.block4)
        #decoder
        self.block5 = []
        self.b5_start = nn.Upsample(scale_factor=2)
        self.block5.append(nn.Conv2d(64 + 32, 32, kernel_size=3, padding=1))
        self.block5.append(nn.ReLU())
        self.block5.append(nn.BatchNorm2d(32))
        self.block5.append(nn.Conv2d(32, 32, kernel_size=3, padding=1))
        self.block5.append(nn.ReLU())
        self.block5.append(nn.BatchNorm2d(32))
        self.b5 = nn.Sequential(*self.block5)

        self.block6 = []
        self.b6_start = nn.Upsample(scale_factor=2)
        self.block6.append(nn.Conv2d(32 + 16, 16, kernel_size=3, padding=1))
        self.block6.append(nn.ReLU())
        self.block6.append(nn.BatchNorm2d(16))
        self.block6.append(nn.Conv2d(16, 16, kernel_size=3, padding=1))
        self.block6.append(nn.ReLU())
        self.block6.append(nn.BatchNorm2d(16))
        self.b6 = nn.Sequential(*self.block6)

        self.block7 = []
        self.b7_start = nn.Upsample(scale_factor=2)
        self.block7.append(nn.Conv2d(16 +8, 8, kernel_size=3, padding=1))
        self.block7.append(nn.ReLU())
        self.block7.append(nn.BatchNorm2d(8))
        self.block7.append(nn.Conv2d(8, 8, kernel_size=3, padding=1))
        self.block7.append(nn.ReLU())
        self.block7.append(nn.BatchNorm2d(8))
        self.b7 = nn.Sequential(*self.block7)

        self.relu = nn.ReLU()
        self.last_layer = nn.Conv2d(8, 1, kernel_size=3, padding=1)

    def forward(self, x):
        
        x1_skip = self.b1(x)
        x1_1 = self.b1_end(x1_skip)

        x2_skip = self.b2(x1_1)
        x2_1 = self.b2_end(x2_skip)
        
        x3_skip = self.b3(x2_1)
        x3_1 = self.b3_end(x3_skip)

        x4 = self.b4(x3_1)

        x5_0 = self.b5_start(x4)
        x5_1 = torch.cat((x5_0, x3_skip), dim=1)
        x5_2 = self.b5(x5_1)

        x6_0 = self.b6_start(x5_2)
        x6_1 = torch.cat((x6_0, x2_skip), dim=1)
        x6_2 = self.b6(x6_1)

        x7_0 = self.b7_start(x6_2)
        x7_1 = torch.cat((x7_0, x1_skip), dim=1)
        x7_2 = self.b7(x7_1)

        out = self.relu(self.last_layer(x7_2))

        return out
    
def EMDenoiseNet(input_shape=(256,256,1)):
    return EMDenoiseNetSkeleton(input_shape)
