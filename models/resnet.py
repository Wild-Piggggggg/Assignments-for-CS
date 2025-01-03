import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic_module import BasicModule

class ResidualBlock(nn.Module):
    """
    implementing submodule: residual block
    """
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super().__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel,outchannel,kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel,outchannel,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut

    def forward(self,x):
        out = self.left(x)
        residue = x if self.right is None else self.right(x)
        out += residue
        return F.relu(out)

class ResNet(BasicModule):
    """
    implementing main module: ResNet34
    """
    def __init__(self, num_classes=2):
        super().__init__()
        # the front few layers for image process
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )

        ### repeate layer of 3 kinds
        self.layer1 = self._make_layer(64,64,3,1,is_shortcut=False)
        self.layer2 = self._make_layer(64, 128, 4, 2)
        self.layer3 = self._make_layer(128, 256, 6, 2)
        self.layer4 = self._make_layer(256, 512, 3, 2)
        ###

        # classification head fc layer
        self.classifier = nn.Linear(512, num_classes)

    def _make_layer(self, inchannel, outchannel, block_num, stride, is_shortcut=True):
        """
        create `layer` including multiple blocks
        """
        if is_shortcut:
            shortcut = nn.Sequential(
                nn.Conv2d(inchannel,outchannel,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(outchannel)
            )
        else:
            shortcut = None

        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))

        for _ in range(1,block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x,7)
        x = x.view(x.size(0), -1)
        return self.classifier(x)