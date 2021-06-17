import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np

__all__ = ['PointNet']


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        return x


class PointNetMLP(nn.Module):
    def __init__(self, num_classes=40, use_normals=False, **kwargs):
        super(PointNet, self).__init__()
        if use_normals:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)), inplace=True)
        x = F.relu(self.bn2(self.dropout(self.fc2(x))), inplace=True)
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    print("===> testing pointNet with use_normals")
    data = torch.rand(10, 6, 1024)
    model = PointNet(k=40, use_normals=True)
    out = model(data)
    print(out.shape)

    print("===> testing pointNet without use_normals")
    data = torch.rand(10, 3, 1024)
    model = PointNet(k=40, use_normals=False)
    out = model(data)
    print(out.shape)
