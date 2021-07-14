"""
PointsformerE2, 1)change relu to GELU, 2) change backbone to model24.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


class LocalGrouper(nn.Module):
    def __init__(self, groups, kneighbors, **kwargs):
        """
        Give xyz[b,p,3] and fea[b,p,d], return new_xyz[b,g,3] and new_fea[b,g,k,2d]
        :param groups: groups number
        :param kneighbors: k-nerighbors
        :param kwargs: others
        """
        super(LocalGrouper, self).__init__()
        self.groups = groups
        self.kneighbors = kneighbors

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        S = self.groups
        xyz = xyz.contiguous()  # xyz [btach, points, xyz]

        fps_idx = torch.multinomial(torch.linspace(0, N - 1, steps=N).repeat(B, 1).to(xyz.device),
                                    num_samples=self.groups, replacement=False).long()
        # fps_idx = farthest_point_sample(xyz, self.groups).long()
        # fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.groups).long() # [B, npoint]
        new_xyz = index_points(xyz, fps_idx)
        new_points = index_points(points, fps_idx)

        idx = knn_point(self.kneighbors, xyz, new_xyz)
        # idx = query_ball_point(radius, nsample, xyz, new_xyz)
        # grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
        grouped_points = index_points(points, idx)
        grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)
        new_points = torch.cat([grouped_points_norm,
                                new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1)]
                               , dim=-1)
        return new_xyz, new_points


class FCBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=False):
        super(FCBNReLU1D, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.net(x)


class FCBNReLU1DRes(nn.Module):
    def __init__(self, channel, kernel_size=1, bias=False):
        super(FCBNReLU1DRes, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(channel),
            nn.GELU(),
            nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(channel)
        )

    def forward(self, x):
        return F.gelu(self.net(x)+x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 32, dropout = 0.):
        super().__init__()

    def forward(self, x):
        return 0


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=32, **kwargs):
        """
        [b batch, d dimension, k points]
        :param dim: input data dimension
        :param heads: heads number
        :param dim_head: dimension in each head
        :param kwargs:
        """
        super(TransformerBlock, self).__init__()
        self.ffn = nn.Sequential(
            nn.Conv1d(dim, dim, 1, bias=False),
            nn.BatchNorm1d(dim)
        )


    def forward(self, x):
        """
        :input x: [b batch, d dimension, p points,]
        :return: [b batch,  d dimension, p points,]
        """
        out = self.ffn(x)
        out = F.gelu(x + out)
        return out


class PreExtraction(nn.Module):
    def __init__(self, channels, blocks=1):
        """
        input: [b,g,k,d]: output:[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PreExtraction, self).__init__()
        operation = []
        for _ in range(blocks):
            operation.append(
                FCBNReLU1DRes(channels)
            )
        self.operation = nn.Sequential(*operation)
        self.transformer = TransformerBlock(channels, heads=4)
    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6])
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        batch_size, _, N = x.size()
        x = self.operation(x)  # [b, d, k]
        x = self.transformer(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x


class PosExtraction(nn.Module):
    def __init__(self, channels, blocks=1):
        """
        input[b,d,g]; output[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PosExtraction, self).__init__()
        operation = []
        for _ in range(blocks):
            operation.append(
                FCBNReLU1DRes(channels)
            )
        self.operation = nn.Sequential(*operation)
        self.transformer = TransformerBlock(channels, heads=4)

    def forward(self, x):  # [b, d, k]
        return self.transformer(self.operation(x))


class encoder_stage(nn.Module):
    def __init__(self, anchor_points=1024, channel=64,
                 pre_blocks=2, pos_blocks=2, k_neighbor=32, reduce=False, **kwargs):
        super(encoder_stage, self).__init__()
        out_channel = channel * 2
        # append local_grouper_list
        self.reduce = reduce
        if self.reduce:
            self.reducer = nn.Sequential(
                nn.Linear(out_channel, channel),
                nn.GELU()
            )
            out_channel = channel
        self.local_grouper = LocalGrouper(anchor_points, k_neighbor)  # [b,g,k,d]
        # append pre_block_list
        self.pre_block_module = PreExtraction(out_channel, pre_blocks)
        # append pos_block_list
        self.pos_block_module = PosExtraction(out_channel, pos_blocks)

    def forward(self, xyz, x):
        xyz, x = self.local_grouper(xyz, x.permute(0, 2, 1)) # [b,g,3]  [b,g,k,d]
        if hasattr(self,"reducer"):
            x = self.reducer(x)
        x = self.pre_block_module(x)  # [b,d,g]
        x = self.pos_block_module(x)  # [b,d,g]
        return xyz, x

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        # xyz1 = xyz1.permute(0, 2, 1)
        # xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.gelu(bn(conv(new_points)))
        return new_points




class PointMLP1(nn.Module):
    def __init__(self, num_classes=50,points=2048, embed_dim=128, normal_channel=True,
                 pre_blocks=[2,2,2,2], pos_blocks=[2,2,2,2], k_neighbors=[32,32,32,32],
                 reducers=[2,2,2,2], **kwargs):
        super(PointMLP1, self).__init__()
        # self.stages = len(pre_blocks)
        self.num_classes = num_classes
        self.points=points
        input_channel=6 if normal_channel else 3
        self.embedding = nn.Sequential(
            FCBNReLU1D(input_channel, embed_dim),
            FCBNReLU1D(embed_dim, embed_dim)
        )

        self.encoder_stage1 = encoder_stage(anchor_points=points//4, channel=128, reduce=False,
                                            pre_blocks=3, pos_blocks=3, k_neighbor=32)
        self.encoder_stage2 = encoder_stage(anchor_points=points//8, channel=256, reduce=True,
                                            pre_blocks=3, pos_blocks=3, k_neighbor=32)
        self.encoder_stage3 = encoder_stage(anchor_points=points // 16, channel=256, reduce=False,
                                            pre_blocks=3, pos_blocks=3, k_neighbor=32)
        self.encoder_stage4 = encoder_stage(anchor_points=points // 32, channel=512, reduce=True,
                                            pre_blocks=3, pos_blocks=3, k_neighbor=32)

        self.fp4 = PointNetFeaturePropagation(in_channel=(512+512), mlp=[512,256,256])
        self.fp3 = PointNetFeaturePropagation(in_channel=256+256, mlp=[512, 256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 256, mlp=[256, 256])
        self.fp1 = PointNetFeaturePropagation(in_channel=256+128+128, mlp=[256, 256])

        self.info_encoder = nn.Sequential(
            FCBNReLU1D(16+3+input_channel, 128),
            FCBNReLU1D(128, 128),
        )
        self.global_encoder = nn.Sequential(
            FCBNReLU1D(512, 256),
            FCBNReLU1D(256, 128),
        )

        self.conv0 = nn.Conv1d(256, 256, 1)
        self.bn0 = nn.BatchNorm1d(256)
        self.drop0 = nn.Dropout(0.4)
        self.conv1 = nn.Conv1d(256, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.4)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, x, norm_plt, cls_label):
        x = torch.cat([x,norm_plt],dim=1)
        points_0 = x
        B, C, N = x.shape
        xyz = x.permute(0, 2, 1)[:,:,:3]
        batch_size, _, _ = x.size()
        x = self.embedding(x) # B,D,N
        xyz_1, fea_1 = self.encoder_stage1(xyz, x)  # [b,p1,3] [b,d1,p1]
        xyz_2, fea_2 = self.encoder_stage2(xyz_1, fea_1)  # [b,p2,3] [b,d2,p2]
        xyz_3, fea_3 = self.encoder_stage3(xyz_2, fea_2)  # [b,p3,3] [b,d3,p3]
        xyz_4, fea_4 = self.encoder_stage4(xyz_3, fea_3)  # [b,p4,3] [b,d4,p3]
        global_context = F.adaptive_max_pool1d(fea_4, 1)

        l3_points = self.fp4(xyz_3, xyz_4, fea_3, fea_4)
        l2_points = self.fp3(xyz_2, xyz_3, fea_2, l3_points)
        l1_points = self.fp2(xyz_1, xyz_2, fea_1, l2_points)
        cls_label_one_hot = cls_label.view(B, 16, 1).repeat(1, 1, N)
        extra_info = torch.cat([cls_label_one_hot, xyz.permute(0, 2, 1), points_0], 1)
        extra_info = self.info_encoder(extra_info)
        global_context = self.global_encoder(global_context)
        l0_points = self.fp1(xyz, xyz_1, torch.cat([extra_info,global_context.expand_as(extra_info) ], 1), l1_points)

        # FC layers
        feat = F.gelu(self.bn0(self.conv0(l0_points)))
        feat = self.drop0(feat)
        feat = F.gelu(self.bn1(self.conv1(feat)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x




class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss

if __name__ == '__main__':
    batch, groups,neighbors,dim=2,512,32,16
    x = torch.rand(batch,groups,neighbors,dim)
    pre_extractor = PreExtraction(dim,3)
    out = pre_extractor(x)
    print(out.shape)

    x = torch.rand(batch, dim, groups)
    pos_extractor = PosExtraction(dim, 3)
    out = pos_extractor(x)
    print(out.shape)


    data = torch.rand(2, 3, 2048)
    norm = torch.rand(2, 3, 2048)
    cls_label = torch.rand([2, 16])
    print("===> testing model ...")
    model = PointMLP1(points=2048)
    out = model(data,norm, cls_label)
    print(out.shape)
