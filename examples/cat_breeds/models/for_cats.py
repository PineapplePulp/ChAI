import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(8192, 256),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.layers(x)
    
    # NOTES
    # Must use nn.Flatten instead of x.view()
    # Must use pool1 and pool2 rather than one pool.