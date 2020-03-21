import torch
import torch.nn as nn
import torch.nn.functional as F
dropout_value = 0.05

class CIFAR10Net_S9(nn.Module):

    def __init__(self):
        super(CIFAR10Net_S9, self).__init__()
        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # in = 32x32x3 , out = 32x32x32, RF = 3

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # in = 32x32x32 , out = 32x32x64, RF = 5
 
        # TRANSITION BLOCK 1
        self.pool4 = nn.MaxPool2d(2, 2) # in = 32x32x64 , out = 16x16x64, RF = 6

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # in = 16x16x64 , out = 16x16x64, RF = 10
        
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)   
        ) # in = 16x16x1x64 , out = 16x16x64, RF = 14
        
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)   
        ) # in = 16x16x1x64 , out = 16x16x64, RF = 18 

        # TRANSITION BLOCK 2
        self.pool8 = nn.MaxPool2d(2, 2) # in = 16x16x64 , out = 8x8x64, RF = 20
    

        # CONVOLUTION BLOCK 3
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # in = 8x8x64 , out = 8x8x32, RF = 28
        
        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # in = 8x8x64 , out = 8x8x64, RF = 36 
        
        self.convblock11 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # in = 8x8x64 , out = 8x8x32, RF = 44              
      
        # OUTPUT BLOCK
        self.Gap1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=8)
        ) # in = 8x8x32 , out = 1x1x32, RF = 72	
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
        ) # in = 1x1x32 , out = 1x1x10, RF = 72

    def forward(self, x):
        x = self.convblock3(self.convblock2(x))
        x = self.pool4(x)
        x = self.convblock7(self.convblock6(self.convblock5(x)))
        x = self.pool8(x)
        x = self.convblock11(self.convblock10(self.convblock9(x)))
        x = self.fc1(self.Gap1(x))
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
