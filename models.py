## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self, n_values=136): #n_classes=10):
        super(Net, self).__init__()

        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        #self.conv1 = nn.Conv2d(1, 32, 5)

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting

        n=32

        self.conv11 = nn.Conv2d(1, n, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(n)
        self.conv12 = nn.Conv2d(n, n, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(n)
        self.mp1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv21 = nn.Conv2d(n, 2*n, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(2*n)
        self.conv22 = nn.Conv2d(2*n, 2*n, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(2*n)
        self.mp2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv31 = nn.Conv2d(2*n, 4*n, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(4*n)
        self.conv32 = nn.Conv2d(4*n, 4*n, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(4*n)
        self.mp3 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv41 = nn.Conv2d(4*n, 8*n, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(8*n)
        self.conv42 = nn.Conv2d(8*n, 8*n, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(8*n)
        self.mp4 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv51 = nn.Conv2d(8*n, 16*n, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm2d(16*n)
        self.conv52 = nn.Conv2d(16*n, 16*n, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(16*n)
        self.mp5 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(16*n, n_values)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))

        # a modified x, having gone through all the layers of your model, should be returned
        out = F.relu( self.bn11( self.conv11(x)))
        out = F.relu( self.bn12( self.conv12(out)))
        out = self.mp1( out)
        out = F.relu( self.bn21( self.conv21(out)))
        out = F.relu( self.bn22( self.conv22(out)))
        out = self.mp2( out)
        out = F.relu( self.bn31( self.conv31(out)))
        out = F.relu( self.bn32( self.conv32(out)))
        out = self.mp3( out)
        out = F.relu( self.bn41( self.conv41(out)))
        out = F.relu( self.bn42( self.conv42(out)))
        out = self.mp4( out)
        out = F.relu( self.bn51( self.conv51(out)))
        out = F.relu( self.bn52( self.conv52(out)))
        out = self.mp5( out)
        out = self.avgpool( out)
        out = out.reshape( out.size(0), -1)
        out = self.fc1(out)

        return out
