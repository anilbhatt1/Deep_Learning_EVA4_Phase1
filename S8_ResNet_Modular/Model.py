import torch
import torch.nn as nn
import torch.nn.functional as F
dropout_value = 0.05

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # CONVOLUTION BLOCK 1
        self.convblock1A = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # in = 32x32x3 , out = 32x32x32, RF = 3

        self.dilated1B = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=2, bias=False, dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # in = 32x32x32 , out = 32x32x64, RF = 7
 
        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # in = 32x32x64 , out = 16x16x64, RF = 8
        self.tran1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), padding=0, bias=False)
        ) # in = 16x16x64 , out = 16x16x32, RF = 6        

        # CONVOLUTION BLOCK 2
        self.convblock2A = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # in = 16x16x32 , out = 16x16x64, RF = 12
        self.depthwise2B = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False, groups=64),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)   
        ) # in = 16x16x1x64 , out = 16x16x64, RF = 16
        self.pointwise2C = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), padding=0, bias=False)
        ) # in = 16x16x64 , out = 16x16x128, RF = 16   

        # TRANSITION BLOCK 2
        self.pool2 = nn.MaxPool2d(2, 2) # in = 16x16x128 , out = 8x8x128, RF = 18
        self.tran2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1), padding=0, bias=False)
        ) # in = 8x8x128 , out = 8x8x32, RF = 18        

        # CONVOLUTION BLOCK 3
        self.convblock3A = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # in = 8x8x32 , out = 8x8x64, RF = 26
        
        # TRANSITION BLOCK 3
        self.pool3 = nn.MaxPool2d(2, 2) # in = 8x8x64 , out = 4x4x64, RF = 30
        self.tran3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), padding=1, bias=False)
        ) # in = 4x4x64 , out = 4x4x32, RF = 30
        
        # OUTPUT BLOCK
        self.Gap1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        ) # in = 4x4x32 , out = 1x1x32, RF = 54	
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
        ) # in = 1x1x32 , out = 1x1x10, RF = 54

    def forward(self, x):
        x = self.dilated1B(self.convblock1A(x))
        x = self.tran1(self.pool1(x))
        x = self.pointwise2C(self.depthwise2B(self.convblock2A(x)))
        x = self.tran2(self.pool2(x))
        x = self.convblock3A(x)
        x = self.tran3(self.pool3(x))
        x = self.fc1(self.Gap1(x))
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
