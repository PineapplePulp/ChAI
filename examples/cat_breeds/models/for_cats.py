import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        # self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.dropout1 = nn.Dropout(0.25)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(8192, 256)
        self.fc2 = nn.Linear(256, 12)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        # x = x.view(-1, 8192)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.fc2(x)
        # x = F.softmax(x, dim=1)
        return x
    
    # NOTES
    # Must use nn.Flatten instead of x.view()
    # Must use pool1 and pool2 rather than one pool.
    # Must not use "same" as padding, because this breaks getInt from moduleAttributes.