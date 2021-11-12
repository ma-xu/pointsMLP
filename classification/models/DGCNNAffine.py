import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def geometric_point_descriptor(x, k=3, idx=None):
    # x: B,3,N
    batch_size = x.size(0)
    num_points = x.size(2)
    org_x = x
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx_base = idx_base.type(torch.cuda.LongTensor)
    idx = idx.type(torch.cuda.LongTensor)
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    neighbors = x.view(batch_size * num_points, -1)[idx, :]
    neighbors = neighbors.view(batch_size, num_points, k, num_dims)

    neighbors = neighbors.permute(0, 3, 1, 2)  # B,C,N,k
    neighbor_1st = torch.index_select(neighbors, dim=-1, index=torch.cuda.LongTensor([1]))  # B,C,N,1
    neighbor_1st = torch.squeeze(neighbor_1st, -1)  # B,3,N
    neighbor_2nd = torch.index_select(neighbors, dim=-1, index=torch.cuda.LongTensor([2]))  # B,C,N,1
    neighbor_2nd = torch.squeeze(neighbor_2nd, -1)  # B,3,N

    edge1 = neighbor_1st - org_x
    edge2 = neighbor_2nd - org_x
    normals = torch.cross(edge1, edge2, dim=1)  # B,3,N
    dist1 = torch.norm(edge1, dim=1, keepdim=True)  # B,1,N
    dist2 = torch.norm(edge2, dim=1, keepdim=True)  # B,1,N

    new_pts = torch.cat((org_x, normals, dist1, dist2, edge1, edge2), 1)  # B,14,N

    return new_pts


class get_graph_feature(nn.Module):
    def __init__(self, dim, k=20, idx=None):
        super(get_graph_feature, self).__init__()
        self.k = k
        self.idx = idx
        self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1,dim]))
        self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, dim]))

    def forward(self, x):
        k=self.k
        idx = self.idx
        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points)
        if idx is None:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        device = torch.device('cuda')

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        idx_base = idx_base.type(torch.cuda.LongTensor)
        idx = idx.type(torch.cuda.LongTensor)
        idx = idx + idx_base
        idx = idx.view(-1)

        _, num_dims, _ = x.size()

        x = x.transpose(2,
                        1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
        feature = x.view(batch_size * num_points, -1)[idx, :]   # [batch, points, dim]
        feature = feature.view(batch_size, num_points, k, num_dims)  # [batch, points,kneighbor, dim]
        # here is the affine operation
        mean = x.unsqueeze(dim=-2)  # [B, npoint, 1, d+3]
        std = torch.std(feature-mean)
        feature = (feature-mean)/(std + 1e-5)
        feature = self.affine_alpha*feature + self.affine_beta
        # print(x.shape, feature.shape)
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

        feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)
        # print(feature.shape)

        return feature


class DGCNNAffine(nn.Module):
    def __init__(self, output_channels=40):
        super(DGCNNAffine, self).__init__()
        self.k = 20

        self.get_graph_feature1 = get_graph_feature(dim=3, k=self.k)
        self.get_graph_feature2 = get_graph_feature(dim=64, k=self.k)
        self.get_graph_feature3 = get_graph_feature(dim=64, k=self.k)
        self.get_graph_feature4 = get_graph_feature(dim=128, k=self.k)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(1024 * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.get_graph_feature1(x)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = self.get_graph_feature2(x1)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = self.get_graph_feature3(x2)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = self.get_graph_feature4(x3)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x


if __name__ == '__main__':
    data = torch.rand(2, 3, 1024)
    model = DGCNNAffine().cuda()
    out = model(data)
    print(out.shape)
