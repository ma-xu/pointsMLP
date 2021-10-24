"""
Based on pformer2, add initial value for residual path in Transformer [see timm Transformer models]

Channel-wise std (std over g and k)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat
from pointnet2_ops import pointnet2_utils


def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    elif activation.lower() == 'leakyrelu0.2':
        return nn.LeakyReLU(inplace=True,negative_slope=0.2)
    else:
        return nn.ReLU(inplace=True)


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
    def __init__(self, channel, groups, kneighbors, use_xyz=True, normalize="center", **kwargs):
        """
        Give xyz[b,p,3] and fea[b,p,d], return new_xyz[b,g,3] and new_fea[b,g,k,d]
        :param groups: groups number
        :param kneighbors: k-nerighbors
        :param kwargs: others
        """
        super(LocalGrouper, self).__init__()
        self.groups = groups
        self.kneighbors = kneighbors
        self.use_xyz = use_xyz
        if normalize is not None:
            self.normalize = normalize.lower()
        else:
            self.normalize = None
        if self.normalize not in ["center", "anchor"]:
            print(f"Unrecognized normalize parameter (self.normalize), set to None. Should be one of [center, anchor].")
            self.normalize = None
        if self.normalize is not None:
            add_channel=3 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(torch.ones([1,1,1,channel + add_channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channel + add_channel]))

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        S = self.groups
        xyz = xyz.contiguous()  # xyz [btach, points, xyz]

        # fps_idx = torch.multinomial(torch.linspace(0, N - 1, steps=N).repeat(B, 1).to(xyz.device), num_samples=self.groups, replacement=False).long()
        # fps_idx = farthest_point_sample(xyz, self.groups).long()
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.groups).long()  # [B, npoint]
        new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]
        new_points = index_points(points, fps_idx)  # [B, npoint, d]

        idx = knn_point(self.kneighbors, xyz, new_xyz)
        # idx = query_ball_point(radius, nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)  # [B, npoint, k, 3]
        grouped_points = index_points(points, idx)  # [B, npoint, k, d]
        if self.use_xyz:
            grouped_points = torch.cat([grouped_points, grouped_xyz],dim=-1)  # [B, npoint, k, d+3]
        if self.normalize is not None:
            if self.normalize =="center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            if self.normalize =="anchor":
                mean = torch.cat([new_points, new_xyz],dim=-1) if self.use_xyz else new_points
                mean = mean.unsqueeze(dim=-2)  # [B, npoint, 1, d+3]
            # instance-wise points std
            s_b, s_g, s_k, s_d = grouped_points.shape
            std = torch.std((grouped_points-mean).reshape(B, s_g*s_k, s_d), dim=1, keepdim=True).unsqueeze(dim=1)
            grouped_points = (grouped_points-mean)/(std + 1e-5)
            grouped_points = self.affine_alpha*grouped_points + self.affine_beta

        new_points = torch.cat([grouped_points, new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1)], dim=-1)
        return new_xyz, new_points


class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, activation='relu'):
        super(ConvBNReLU1D, self).__init__()
        self.act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)


class ConvBNReLURes1D(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(ConvBNReLURes1D, self).__init__()
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=int(channel * res_expansion),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(channel),
                self.act,
                nn.Conv1d(in_channels=channel, out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel)
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


class PreExtraction(nn.Module):
    def __init__(self, channels, out_channels,  blocks=1, groups=1, res_expansion=1, bias=True,
                 activation='relu', use_xyz=True):
        """
        input: [b,g,k,d]: output:[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PreExtraction, self).__init__()
        in_channels = 3+2*channels if use_xyz else 2*channels
        self.transfer = ConvBNReLU1D(in_channels, out_channels, bias=bias, activation=activation)
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(out_channels, groups=groups, res_expansion=res_expansion,
                                bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6])
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        x = self.transfer(x)
        batch_size, _, _ = x.size()
        x = self.operation(x)  # [b, d, k]
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=8, head_dim=32, similarity="dot"):
        super().__init__()
        self.similarity = similarity
        inner_dim = head_dim * heads
        # project_out = not (heads == 1 and head_dim == dim)
        self.heads = heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.attend = nn.Softmax(dim=-1)
        # self.to_q = nn.Conv1d(dim, inner_dim, 1, groups=heads, bias=False)
        # self.to_k = nn.Conv1d(dim, inner_dim, 1, groups=heads, bias=False)
        # self.to_v = nn.Conv1d(dim, inner_dim, 1, groups=heads, bias=False)
        self.to_qkv = nn.Conv1d(dim, inner_dim * 3, 1, bias=False)

        self.to_out = nn.Conv1d(inner_dim, dim, 1)

    def forward(self, x):  # [b,d,k]
        b, _, n, h = *x.shape, self.heads
        # q = self.to_q(x)
        # k = self.to_k(x)
        # v = self.to_v(x)
        # q, k, v = map(lambda t: rearrange(t, 'b (h d) n -> b h n d', h=h), [q, k, v])
        qkv = self.to_qkv(x).permute(0, 2, 1).chunk(3, dim=-1)  # [b, k, h*d]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # [b,h,k,d]
        if self.similarity == "l2":
            smi = q.unsqueeze(dim=-2) - k.unsqueeze(dim=-3)
            smi = -torch.norm(smi, p=2, dim=-1, keepdim=False) * self.scale
        elif self.similarity == "l1":
            smi = q.unsqueeze(dim=-2) - k.unsqueeze(dim=-3)
            smi = -torch.norm(smi, p=1, dim=-1, keepdim=False) * self.scale
        else:
            smi = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(smi)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, head_dim=64, activation='relu', similarity="dot", bias=True, res_expansion=1.0, init_values=1e-4, **kwargs):
        """
        x [b*g, d, k]  aug_xyz[b,g,k,dim]
        :input x: [b batch, d dimension, p points,]
        :return: [b batch,  d dimension, p points,]
        """
        super(TransformerBlock, self).__init__()
        self.attention = Attention(dim=dim, heads=heads, head_dim=head_dim, similarity=similarity)
        self.ffn = nn.Sequential(
            ConvBNReLU1D(dim, int(dim * res_expansion), activation=activation, bias=bias),
            nn.Conv1d(in_channels=int(dim * res_expansion), out_channels=dim, kernel_size=1, bias=bias)
        )
        self.act = get_activation(activation)

        self.ls1 = nn.Parameter(init_values * torch.ones(1,dim,1))
        self.ls2 = nn.Parameter(init_values * torch.ones(1,dim,1))

    def forward(self, x):  # x [b*g, d, k]  aug_xyz[b,g,k,heads]
        att = self.attention(x)  # x+position[b*g, d, k]   att[b*g, d, k]
        att = self.ls1*att + x
        out = self.ffn(att)
        out = self.act(self.ls2*out + att)
        return out


class PosExtraction(nn.Module):
    def __init__(self, channels, blocks=1, groups=1, res_expansion=1, bias=True, activation='relu',
                 heads=8, head_dim=64, similarity="dot", use_xyz=True):
        """
        input[b,d,g]; output[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PosExtraction, self).__init__()
        self.use_xyz = use_xyz
        if self.use_xyz:
            self.position_embed = nn.Sequential(
                ConvBNReLU1D(3, int(channels*res_expansion), bias=bias, activation=activation),
                ConvBNReLU1D(int(channels*res_expansion), channels, bias=bias, activation=activation)
            )
        operation = []
        for _ in range(blocks):
            operation.append(
                TransformerBlock(dim=channels, heads=heads, head_dim=head_dim, activation=activation,
                                 similarity=similarity, bias=bias, res_expansion=res_expansion)
                # ConvBNReLURes1D(channels, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x, xyz):  # [b, d, g], [b,g,3]
        if self.use_xyz:
            mean,std = torch.std_mean(xyz, dim=1, keepdim=True)
            xyz = (xyz-mean)/( std + 1e-5)
            x += self.position_embed(xyz.permute(0, 2, 1))
        return self.operation(x)


class Model(nn.Module):
    def __init__(self, points=1024, class_num=40, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=True, use_xyz=True, normalize="center",
                 dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 heads = [1,1,1,1], head_dims = [64,64,64,64], similarity="dot",
                 k_neighbors=[32, 32, 32, 32], reducers=[2, 2, 2, 2], **kwargs):
        super(Model, self).__init__()
        self.stages = len(pre_blocks)
        self.class_num = class_num
        self.points = points
        self.embedding = ConvBNReLU1D(3, embed_dim, bias=bias, activation=activation)
        assert len(pre_blocks) == len(k_neighbors) == len(reducers) == len(pos_blocks) == len(dim_expansion), \
            "Please check stage number consistent for pre_blocks, pos_blocks k_neighbors, reducers."
        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()
        last_channel = embed_dim
        anchor_points = self.points
        for i in range(len(pre_blocks)):
            out_channel = last_channel * dim_expansion[i]
            pre_block_num = pre_blocks[i]
            pos_block_num = pos_blocks[i]
            kneighbor = k_neighbors[i]
            reduce = reducers[i]
            anchor_points = anchor_points // reduce
            # append local_grouper_list
            local_grouper = LocalGrouper(last_channel, anchor_points, kneighbor, use_xyz, normalize)  # [b,g,k,d]
            self.local_grouper_list.append(local_grouper)
            # append pre_block_list
            pre_block_module = PreExtraction(last_channel, out_channel, pre_block_num, groups=groups,
                                             res_expansion=res_expansion,
                                             bias=bias, activation=activation, use_xyz=use_xyz)
            self.pre_blocks_list.append(pre_block_module)
            # append pos_block_list
            pos_block_module = PosExtraction(out_channel, pos_block_num, groups=groups, use_xyz=use_xyz,
                                             res_expansion=res_expansion, bias=bias, activation=activation,
                                             heads=heads[i], head_dim=head_dims[i], similarity=similarity)
            self.pos_blocks_list.append(pos_block_module)

            last_channel = out_channel

        self.act = get_activation(activation)
        self.classifier = nn.Sequential(
            nn.Linear(last_channel, 512),
            nn.BatchNorm1d(512),
            self.act,
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            self.act,
            nn.Dropout(0.5),
            nn.Linear(256, self.class_num)
        )

    def forward(self, x):
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        x = self.embedding(x)  # B,D,N
        for i in range(self.stages):
            # Give xyz[b, p, 3] and fea[b, p, d], return new_xyz[b, g, 3] and new_fea[b, g, k, d]
            xyz, x = self.local_grouper_list[i](xyz, x.permute(0, 2, 1))  # [b,g,3]  [b,g,k,d]
            x = self.pre_blocks_list[i](x)  # [b,d,g]
            x = self.pos_blocks_list[i](x,xyz)  # [b,d,g]

        x = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)
        x = self.classifier(x)
        return x


def pformer3A(num_classes=40, **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=64, groups=1, res_expansion=1.0,
                   activation="relu", bias=True, use_xyz=False, normalize="anchor",
                   dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[1,1,1,1],
                   heads = [1,1,1,1], head_dims = [1,1,1,1], similarity="dot",
                   k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2], **kwargs)


def pformer3B(num_classes=40, **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=64, groups=1, res_expansion=1.0,
                   activation="gelu", bias=True, use_xyz=False, normalize="anchor",
                   dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[1,1,1,1],
                   heads = [1,1,1,1], head_dims = [1,1,1,1], similarity="dot",
                   k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2], **kwargs)


def pformer3C(num_classes=40, **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=64, groups=1, res_expansion=1.0,
                   activation="leakyrelu0.2", bias=True, use_xyz=False, normalize="anchor",
                   dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[1,1,1,1],
                   heads = [1,1,1,1], head_dims = [1,1,1,1], similarity="dot",
                   k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2], **kwargs)


def pformer3D(num_classes=40, **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=64, groups=1, res_expansion=1.0,
                   activation="gelu", bias=True, use_xyz=True, normalize="anchor",
                   dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[1,1,1,1],
                   heads = [1,1,1,1], head_dims = [1,1,1,1], similarity="dot",
                   k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2], **kwargs)

def pformer3E(num_classes=40, **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=64, groups=1, res_expansion=1.0,
                   activation="gelu", bias=False, use_xyz=True, normalize="anchor",
                   dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[1,1,1,1],
                   heads = [1,1,1,1], head_dims = [1,1,1,1], similarity="dot",
                   k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2], **kwargs)


def pformer3F(num_classes=40, **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=64, groups=1, res_expansion=1.0,
                   activation="gelu", bias=False, use_xyz=True, normalize="center",
                   dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[1,1,1,1],
                   heads = [1,1,1,1], head_dims = [1,1,1,1], similarity="dot",
                   k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2], **kwargs)


def pformer3G(num_classes=40, **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=64, groups=1, res_expansion=1.0,
                   activation="relu", bias=True, use_xyz=False, normalize="anchor",
                   dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2,2,2,2],
                   heads = [1,1,1,1], head_dims = [1,1,1,1], similarity="dot",
                   k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2], **kwargs)

def pformer3H(num_classes=40, **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=64, groups=1, res_expansion=1.0,
                   activation="relu", bias=True, use_xyz=False, normalize="anchor",
                   dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[1,1,1,1],
                   heads = [6,6,6,6], head_dims = [1,1,1,1], similarity="dot",
                   k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2], **kwargs)

def pformer3I(num_classes=40, **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=64, groups=1, res_expansion=1.0,
                   activation="relu", bias=True, use_xyz=False, normalize="anchor",
                   dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[1,1,1,1],
                   heads = [1,1,1,1], head_dims = [64,64,64,64], similarity="dot",
                   k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2], **kwargs)

def pformer3J(num_classes=40, **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=64, groups=1, res_expansion=1.0,
                   activation="relu", bias=True, use_xyz=False, normalize="anchor",
                   dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[1,1,1,1],
                   heads = [6,6,6,6], head_dims = [64,64,64,64], similarity="dot",
                   k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2], **kwargs)

if __name__ == '__main__':
    data = torch.rand(2, 3, 1024)
    print("===> testing model ...")
    model = pformer3A()
    out = model(data)
    print(out.shape)

    model = pformer3B()
    out = model(data)
    print(out.shape)

    model = pformer3C()
    out = model(data)
    print(out.shape)

    model = pformer3D()
    out = model(data)
    print(out.shape)

