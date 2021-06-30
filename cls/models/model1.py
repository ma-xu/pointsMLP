"""
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat


# from pointnet2_ops import pointnet2_utils


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


def get_pool_func(pool="max"):
    if pool == "max":
        pool_func = torch.amax
    elif pool == "mean":
        pool_func = torch.mean
    elif pool == "logsumexp":
        pool_func = torch.logsumexp
    elif pool == "norm":
        pool_func = torch.norm
    return pool_func


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
        xyz = xyz.contiguous()  # xyz [btach, points, xyz]

        fps_idx = farthest_point_sample(xyz, self.groups).long()
        # fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.groups).long()  # [B, npoint]
        new_xyz = index_points(xyz, fps_idx)
        new_points = index_points(points, fps_idx)

        idx = knn_point(self.kneighbors, xyz, new_xyz)
        # idx = query_ball_point(radius, nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
        grouped_points = index_points(points, idx)

        return new_xyz, new_points, grouped_xyz, grouped_points  # [b,p,3] [b, p, d] [b,p,k,3] [b,p,k,d]


class PositionalEmbedding(nn.Module):
    def __init__(self, channel):
        super(PositionalEmbedding, self).__init__()
        self.channel = channel
        self.embedding = nn.Sequential(
            nn.Linear(9, max(self.channel // 4, 16)),
            nn.LeakyReLU(inplace=True),
            nn.Linear(max(self.channel // 4, 16), self.channel)
        )

    def forward(self, new_xyz, grouped_xyz):  # [b,p,3] [b,p,k,3]
        relative_position = grouped_xyz - new_xyz.unsqueeze(dim=2)
        std_, mean_ = torch.std_mean(relative_position, dim=2, keepdim=True)
        postion_ = torch.cat([relative_position,
                              std_.expand_as(relative_position), mean_.expand_as(relative_position)],
                             dim=-1)
        postion_ = self.embedding(postion_)  # [b, p, k, d]
        return postion_


class SimplePositionalEmbedding(nn.Module):
    def __init__(self, channel):
        super(SimplePositionalEmbedding, self).__init__()
        self.channel = channel
        self.embedding = nn.Sequential(
            nn.Linear(9, max(self.channel // 8, 16)),
            nn.LeakyReLU(inplace=True),
            nn.Linear(max(self.channel // 8, 16), self.channel)
        )

    def forward(self, xyz):  # [b,p,3]
        std_, mean_ = torch.std_mean(xyz, dim=1, keepdim=True)
        postion_ = torch.cat([xyz, std_.expand_as(xyz), mean_.expand_as(xyz)], dim=-1)
        postion_ = self.embedding(postion_)  # [b, p, d]
        return postion_


class ContextAttention(nn.Module):
    def __init__(self, dim, inner_dim, group=1, pool="max"):
        super().__init__()
        self.to_q = nn.Conv2d(dim, inner_dim, 1, groups=group, bias=False)
        self.to_k = nn.Conv2d(dim, inner_dim, 1, groups=group, bias=False)
        self.to_v = nn.Conv2d(dim, inner_dim, 1, groups=group, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1, groups=group, bias=False)
        self.pool_func = get_pool_func(pool)

    def forward(self, points):  # [b,p,k,d]
        context = self.pool_func(points, dim=-2, keepdim=True)  # [b,p,1,d]
        context = context.permute(0, 3, 1, 2)
        points = points.permute(0, 3, 1, 2)  # [b,d,p,k]
        key = self.to_k(context)
        query = self.to_q(points)
        value = self.to_v(points)
        attention = torch.sigmoid(key * query)
        out = attention * value
        out = self.to_out(out).permute(0, 2, 3, 1)  # [0,2,3,1]
        return out


class Transformer(nn.Module):
    def __init__(self, dim, inner_dim, group=1, pool="max", **kwargs):
        super(Transformer, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.attention = ContextAttention(dim=dim, inner_dim=inner_dim, group=group, pool=pool)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(dim, dim)
        )

    def forward(self, points):  # [b,p,k,d]
        att = self.attention(points)  # [b,p,k,d]
        points = att + points
        out = self.ffn(points)
        out = out + points
        return out


class LocalExtraction(nn.Module):
    def __init__(self, dim, inner_dim, group=1, pool="max", blocks=3, **kwargs):
        super(LocalExtraction, self).__init__()
        operation = []
        for _ in range(blocks):
            operation.append(
                Transformer(dim, inner_dim, group, pool)
            )
        self.operation = nn.Sequential(*operation)
        self.pool_func = get_pool_func(pool)
        self.up = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):  # [b,p,k,d]
        out = self.operation(x)  # [b,p,k,d]
        out = self.pool_func(out, dim=-2, keepdim=False)  # [b,p,d]
        out = self.up(out)
        return out


class Attention(nn.Module):
    def __init__(self, dim, heads=8, down=4):
        super().__init__()
        inner_dim = dim // down
        self.heads = heads
        self.scale = (inner_dim // heads) ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, 1)
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TransformerGlobal(nn.Module):
    def __init__(self, dim, heads=8, down=2, **kwargs):
        super(TransformerGlobal, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.attention = Attention(dim, heads, down)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(dim, dim)
        )

    def forward(self, points):  # [b,p,d]
        att = self.attention(points)  # [b,p,d]
        points = att + points
        out = self.ffn(points)
        out = out + points
        return out


class Model1(nn.Module):
    def __init__(self, points=1024, class_num=40, embed_dim=64, pool="max",
                 k_neighbors=[32, 32, 32, 32], expansion=2, groups=8,
                 local_blocks=[2, 2, 2, 2], global_blocks=3,
                 reducers=[2, 2, 2, 2], **kwargs):
        super(Model1, self).__init__()
        assert pool in ["max", 'mean', "logsumexp", "norm"], "Key context must be in ['max', 'mean','logsumexp','norm']"
        self.pool = pool
        self.groups = groups
        self.expansion = expansion
        self.stages = len(local_blocks)
        self.class_num = class_num
        self.points = points
        self.embedding = nn.Linear(3, embed_dim)
        assert len(local_blocks) == len(reducers) == len(k_neighbors), \
            "Please check stage number consistent for local_blocks, k_neighbors, reducers."
        self.local_grouper_list = nn.ModuleList()
        self.positional_embedding_list = nn.ModuleList()
        self.local_blocks_list = nn.ModuleList()
        last_channel = embed_dim
        anchor_points = self.points
        for i in range(len(local_blocks)):
            local_block_num = local_blocks[i]
            kneighbor = k_neighbors[i]
            reduce = reducers[i]
            anchor_points = anchor_points // reduce
            # append local_grouper_list
            local_grouper = LocalGrouper(anchor_points, kneighbor)  # [b,g,k,d]
            self.local_grouper_list.append(local_grouper)
            # append positional_embedding_list
            positional_embedding = PositionalEmbedding(channel=last_channel)
            self.positional_embedding_list.append(positional_embedding)
            # append local_blocks_list
            Local_extraction = LocalExtraction(last_channel, last_channel * self.expansion,
                                               self.groups, self.pool, local_block_num)
            self.local_blocks_list.append(Local_extraction)
            last_channel = last_channel * 2

        global_blocks_list = []
        for _ in range(global_blocks):
            transformer_global = TransformerGlobal(last_channel, self.groups, down=2)
            global_blocks_list.append(transformer_global)
        self.global_transformers = nn.Sequential(*global_blocks_list)
        self.global_positional_embedding = SimplePositionalEmbedding(last_channel)
        self.poolfunc = get_pool_func(pool)
        self.classifier = nn.Sequential(
            nn.Linear(last_channel, last_channel // 4),
            nn.Dropout(0.5),
            nn.Linear(last_channel // 4, self.class_num)
        )

    def forward(self, x):  # [b,3,n]
        xyz = x.permute(0, 2, 1)
        points = self.embedding(xyz)  # [b,n,d]
        # hierarchical local-feature extraction
        for i in range(self.stages):
            # [b,p,3] [b, p, d] [b,p,k,3] [b,p,k,d]
            xyz, _, grouped_xyz, grouped_points = self.local_grouper_list[i](xyz, points)
            position = self.positional_embedding_list[i](xyz, grouped_xyz)  # [b,p,k,d]
            points = self.local_blocks_list[i](grouped_points + position)  # [b,p,d]
        # global-feature extraction
        global_position = self.global_positional_embedding(xyz)
        points = self.global_transformers(points + global_position)
        out = self.poolfunc(points, dim=-2, keepdim=False)
        x = self.classifier(out)
        return x


def model1A(num_classes=40, **kwargs) -> Model1:
    return Model1(points=1024, class_num=num_classes, embed_dim=64, pool="max",
                 k_neighbors=[32, 32], expansion=2, groups=8,
                 local_blocks=[2, 2], global_blocks=3,
                 reducers=[4, 4], **kwargs)

def model1B(num_classes=40, **kwargs) -> Model1:
    return Model1(points=1024, class_num=num_classes, embed_dim=64, pool="mean",
                 k_neighbors=[32, 32], expansion=2, groups=8,
                 local_blocks=[2, 2], global_blocks=3,
                 reducers=[4, 4], **kwargs)

def model1C(num_classes=40, **kwargs) -> Model1:
    return Model1(points=1024, class_num=num_classes, embed_dim=64, pool="logsumexp",
                 k_neighbors=[32, 32], expansion=2, groups=8,
                 local_blocks=[2, 2], global_blocks=3,
                 reducers=[4, 4], **kwargs)

def model1D(num_classes=40, **kwargs) -> Model1:
    return Model1(points=1024, class_num=num_classes, embed_dim=128, pool="max",
                 k_neighbors=[32, 32], expansion=2, groups=8,
                 local_blocks=[2, 2], global_blocks=3,
                 reducers=[4, 4], **kwargs)

def model1E(num_classes=40, **kwargs) -> Model1:
    return Model1(points=1024, class_num=num_classes, embed_dim=128, pool="max",
                 k_neighbors=[64, 64], expansion=2, groups=8,
                 local_blocks=[2, 2], global_blocks=3,
                 reducers=[4, 4], **kwargs)

def model1F(num_classes=40, **kwargs) -> Model1:
    return Model1(points=1024, class_num=num_classes, embed_dim=128, pool="max",
                 k_neighbors=[32, 32], expansion=2, groups=8,
                 local_blocks=[4, 4], global_blocks=3,
                 reducers=[4, 4], **kwargs)

if __name__ == '__main__':
    # test positional encoding
    positional_embedding = PositionalEmbedding(channel=64)
    new_xyz = torch.rand(2, 16, 3)
    grouped_xyz = torch.rand(2, 16, 32, 3)
    out = positional_embedding(new_xyz, grouped_xyz)

    # test Transformer  [b,p,1,d] [b,p,k,d]
    transformer = Transformer(dim=64, inner_dim=256, group=8, pool="norm")
    # context = torch.rand(2, 3, 1, 64)
    grouped_points = torch.rand(2, 3, 16, 64)
    out = transformer(grouped_points)
    print(out.shape)

    # test context attention  [b,p,1,d] [b,p,k,d]
    attention = ContextAttention(dim=64, inner_dim=256, group=8, pool="norm")
    # context = torch.rand(2, 3, 1, 64)
    grouped_points = torch.rand(2, 3, 16, 64)
    out = attention(grouped_points)
    print(out.shape)

    # test LocalExtraction
    local_extraction = LocalExtraction(dim=64, inner_dim=256, group=8, pool="max", blocks=2)
    # context = torch.rand(2, 3, 1, 64)
    grouped_points = torch.rand(2, 3, 16, 64)
    out = local_extraction(grouped_points)
    print(out.shape)

    # test attention  (global) [b,p,d]
    attention = Attention(dim=256, heads=8, down=2)
    points = torch.rand(2, 3, 256)
    out = attention(points)
    print(out.shape)

    data = torch.rand(2, 10, 128)
    att = Attention(128)
    out = att(data)
    print(out.shape)

    data = torch.rand(2, 3, 1024)
    print("===> testing model ...")
    model = Model1()
    out = model(data)
    print(out.shape)

    print("===> testing modelA ...")
    model = model1A()
    out = model(data)
    print(out.shape)

    print("===> testing modelB ...")
    model = model1B()
    out = model(data)
    print(out.shape)

    print("===> testing modelC ...")
    model = model1C()
    out = model(data)
    print(out.shape)

    print("===> testing modelD ...")
    model = model1D()
    out = model(data)
    print(out.shape)

    print("===> testing modelE ...")
    model = model1E()
    out = model(data)
    print(out.shape)

    print("===> testing modelF ...")
    model = model1F()
    out = model(data)
    print(out.shape)
