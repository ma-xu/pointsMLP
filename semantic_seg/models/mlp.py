import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np

__all__ = ['MLP']

class MLP(nn.Module):
    def __init__(self, num_classes=13, channels=9, **kwargs):
        super(MLP, self).__init__()
        self.num_classes = num_classes
        self.channels = channels
        self.net=nn.Sequential(
            nn.Linear(self.channels,64),
            nn.LayerNorm([64]),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.LayerNorm([64]),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.LayerNorm([128]),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.LayerNorm([128]),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.num_classes)
        )



    def forward(self, x):

        return self.net(x).permute(0,2,1)

if __name__ == '__main__':
    x= torch.rand(32,4096,9)
    model = MLP()
    out = model(x)
    print(out.shape)
