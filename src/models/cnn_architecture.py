import torch.nn as nn
from torch import flatten

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()

        # Conv layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.BN1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.BN2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.BN3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        # Pools
        self.pool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # FC layers
        self.fc1 = nn.Linear(128, 64)
        self.relufc1 = nn.ReLU()

        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        # Conv
        z1 = self.conv1(x)
        a1 = self.BN1(z1)
        a1 = self.relu1(a1)
        a1 = self.pool(a1)

        z2 = self.conv2(a1)
        a2 = self.BN2(z2)
        a2 = self.relu2(a2)
        a2 = self.pool(a2)

        z3 = self.conv3(a2)
        a3 = self.BN3(z3)
        a3 = self.relu3(a3)
        a3 = self.pool(a3)

        # Transition
        a4 = self.avgpool(a3)
        a4 = flatten(a4, 1)

        # Fc
        z5 = self.fc1(a4)
        a5 = self.relufc1(z5)
        a5 = self.dropout(a5)
        y = self.fc2(a5)

        return y
