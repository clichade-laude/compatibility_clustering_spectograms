import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, channels=3):
        if channels != 1 and channels != 3:
            raise Exception("Input channels must be 1 or 3")
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=16, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=0)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=0)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(4608, 32)
        self.linear2 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = torch.relu(self.conv2(x))
        x = self.maxpool(x)
        x = torch.relu(self.conv3(x))
        x = self.maxpool(x)
        x = torch.relu(self.conv4(x))
        x = self.flatten(x)
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        x = torch.softmax(x, dim=1)

        return x