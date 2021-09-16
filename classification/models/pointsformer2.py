"""
Exactly based on Model10, but ReLU to GeLU
Based on Model8, add dropout and max, avg combine.
Based on Local model, add residual connections.
The extraction is doubled for depth.
Learning Point Cloud with Progressively Local representation.
[B,3,N] - {[B,G,K,d]-[B,G,d]}  - {[B,G',K,d]-[B,G',d]} -cls
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch import einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from pointsformer_utils import get_activation, square_distance, index_points, \
    farthest_point_sample, query_ball_point, knn_point


from pointnet2_ops import pointnet2_utils

class LocalGrouper(nn.Module):
    def __init__(self, groups, kneighbors, norm_augmented=True, concat_anchor=True, **kwargs):
        """
        Give xyz[b,p,3] and fea[b,p,d], return new_xyz[b,g,3], augmented_xyz [b,g,k,10], and new_fea[b,g,k,(2)d]
        :param groups: groups number
        :param kneighbors: k-nerighbors
        :param kwargs: others
        """
        super(LocalGrouper, self).__init__()
        self.groups = groups
        self.kneighbors = kneighbors
        self.norm_augmented = norm_augmented
        self.concat_anchor = concat_anchor

    def forward(self, xyz, points):
        # B, N, C = xyz.shape
        # S = self.groups
        xyz = xyz.contiguous()  # xyz [btach, points, xyz]
        # fps_idx = farthest_point_sample(xyz, self.groups).long()
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.groups).long() # [B, npoint]
        new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]
        new_points = index_points(points, fps_idx)  # [B, npoint, d]

        idx = knn_point(self.kneighbors, xyz, new_xyz)  # [2, 512, 32]
        # idx = query_ball_point(radius, nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, 3]
        grouped_points = index_points(points, idx)  # [b, npoint, neighbors, dim]
        # augmented positions [b,n,k,10]
        abs_dis = torch.norm(grouped_xyz - new_xyz.unsqueeze(dim=2), p="fro", dim=-1, keepdim=True)  # [b, n, k, d]
        val_dis = grouped_xyz - new_xyz.unsqueeze(dim=2)  # [b, n, k, d]
        augmented_xyz = torch.cat([abs_dis, val_dis,
                                   new_xyz.unsqueeze(dim=2).repeat(1, 1, self.kneighbors, 1),
                                   grouped_xyz], dim=-1)
        if self.norm_augmented:
            augmented_xyz = (augmented_xyz - augmented_xyz.mean(dim=2, keepdim=True)) \
                            / (augmented_xyz.std(dim=2, keepdim=True) + 1e-8)
        if self.concat_anchor:
            grouped_points = torch.cat([grouped_points, new_points.unsqueeze(dim=2).repeat(1, 1, self.kneighbors, 1)],
                                       dim=-1)
        return new_xyz, augmented_xyz, grouped_points


class FCBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, activation="relu"):
        """
        input [b, dim, npoints]
        """
        super(FCBNReLU1D, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            get_activation(activation)
        )

    def forward(self, x):  # [b, dim, npoints]
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, head_dim=32, similarity="dot", dropout=0.):
        super().__init__()
        self.similarity = similarity
        inner_dim = head_dim * heads
        # project_out = not (heads == 1 and head_dim == dim)
        self.heads = heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Conv1d(dim, inner_dim, 1, groups=heads, bias=False)
        self.to_k = nn.Conv1d(dim, inner_dim, 1, groups=heads, bias=False)
        self.to_v = nn.Conv1d(dim, inner_dim, 1, groups=heads, bias=False)
        # self.to_qkv = nn.Conv1d(dim, inner_dim * 3, 1, groups=32, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv1d(inner_dim, dim, 1, groups=heads),
            nn.BatchNorm1d(dim)
        )

    def forward(self, x):  # [b,d,k]
        b, _, n, h = *x.shape, self.heads
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) n -> b h n d', h=h), [q, k, v])
        # qkv = self.to_qkv(x).permute(0,2,1).chunk(3, dim=-1)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        if self.similarity == "l2":
            smi = q.unsqueeze(dim=-2) - k.unsqueeze(dim=-3)
            smi = -torch.norm(smi, p="fro", dim=-1, keepdim=False) * self.scale
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
    def __init__(self, dim, heads=8, head_dim=64, activation='relu', similarity="dot", ffn_ratio=1.0, **kwargs):
        """
        x [b*g, d, k]  aug_xyz[b,g,k,dim]
        :input x: [b batch, d dimension, p points,]
        :return: [b batch,  d dimension, p points,]
        """
        super(TransformerBlock, self).__init__()
        self.attention = Attention(dim=dim, heads=heads, head_dim=head_dim, similarity=similarity)
        self.ffn = nn.Sequential(
            FCBNReLU1D(dim, int(dim * ffn_ratio), activation=activation),
            FCBNReLU1D(int(dim * ffn_ratio), dim, activation=activation)
        )
        self.act = get_activation(activation)

    def forward(self, x, aug_xyz):  # x [b*g, d, k]  aug_xyz[b,g,k,heads]
        att = self.attention(x + aug_xyz)  # x+position[b*g, d, k]   att[b*g, d, k]
        att = self.act(att + x)
        out = self.ffn(att)
        out = self.act(att + out)
        return out


class PreExtraction(nn.Module):
    def __init__(self, in_channel, channels, blocks=1, heads=12, head_dim=64,
                 activation="relu", similarity="dot", ffn_ratio=1.0):
        """
        input: fea[b,g,k,d] aug_xyz[b,g,k,10]: output:[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PreExtraction, self).__init__()
        self.fc = FCBNReLU1D(in_channel, channels, activation=activation)
        operations = nn.ModuleList()
        for _ in range(blocks):
            operations.append(
                TransformerBlock(channels, heads=heads, head_dim=head_dim,
                                 activation=activation, similarity=similarity, ffn_ratio=ffn_ratio)
            )
        # self.operation = nn.Sequential(*operation)
        self.operations = operations

        self.xyz_expand = 4
        xyz_channel = 10
        self.postion_embedding = nn.Sequential(
            nn.Linear(xyz_channel, xyz_channel * self.xyz_expand),
            get_activation(activation),
            nn.Linear(xyz_channel * self.xyz_expand, channels),
            Rearrange('b g k d -> (b g) d k')
        )

    def forward(self, x, augument_xyz):  # input: x[b,g,k,d] aug_xyz[b,g,k,10]: output:[b,d,g]
        b, g, k, d = x.size()  # torch.Size([32, 512, 32, 6])
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, k)
        batch_size, _, _ = x.size()
        x = self.fc(x)
        augument_xyz = self.postion_embedding(augument_xyz)
        for op in self.operations:
            x = op(x, augument_xyz)

        # x = self.operation(x, augument_xyz)  # [b, d, k]
        # x = self.transformer(x, augument_xyz)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, g, -1).permute(0, 2, 1)
        return x


class PosExtraction(nn.Module):
    def __init__(self, channels, blocks=1, heads=8, head_dim=64, activation='relu', similarity="dot",
                 ffn_ratio=1.0, norm_augmented=True):
        """
        input[b,d,g] [b,d,3]; output[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PosExtraction, self).__init__()
        operations = nn.ModuleList()
        for _ in range(blocks):
            operations.append(
                TransformerBlock(channels, heads=heads, head_dim=head_dim, activation=activation, similarity=similarity,
                                 ffn_ratio=ffn_ratio)
            )

        self.operations = operations
        self.xyz_alpha = Parameter(torch.ones([1, 1, 3]))
        self.xyz_beta = Parameter(torch.zeros([1, 1, 3]))
        self.norm_augmented = norm_augmented

        self.xyz_expand = 4
        xyz_channel = 10
        self.postion_embedding = nn.Sequential(
            nn.Linear(xyz_channel, xyz_channel * self.xyz_expand),
            get_activation(activation),
            nn.Linear(xyz_channel * self.xyz_expand, channels),
            Rearrange('b g k d -> (b g) d k')
        )

    def forward(self, x, xyz):  # [b, d, g]  [b, g, 3]
        xyz_mean, xyz_std = torch.std_mean(xyz, dim=1, keepdim=True)
        normed_xyz = (xyz - xyz_mean) / (xyz_std + 1e-8)
        augument_xyz = torch.cat([xyz,
                                  normed_xyz,
                                  torch.norm(xyz, dim=-1, keepdim=True),
                                  torch.cos(self.xyz_alpha * xyz + self.xyz_beta)
                                  ], dim=-1)  # aug_xyz [b,g,10]
        if self.norm_augmented:
            augument_xyz = (augument_xyz - augument_xyz.mean(dim=2, keepdim=True)) \
                           / (augument_xyz.std(dim=2, keepdim=True) + 1e-8)
        augument_xyz = self.postion_embedding(augument_xyz.unsqueeze(dim=1))
        for op in self.operations:
            x = op(x, augument_xyz)
        return x


class Pointsformer2(nn.Module):
    def __init__(self, points=1024, class_num=40, embed_dim=32, activation='relu', smilarity='dot', ffn_ratio=0.125,
                 heads=[16, 16, 16, 16], head_dims=[64, 64, 64, 64], norm_augmented=True, concat_anchor=True,
                 expansions=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[2, 2, 2, 2], **kwargs):
        super(Pointsformer2, self).__init__()
        self.stages = len(pre_blocks)
        self.class_num = class_num
        self.points = points
        self.embedding = nn.Sequential(
            FCBNReLU1D(3, embed_dim, activation=activation),
            FCBNReLU1D(embed_dim, embed_dim, activation=activation)
        )
        assert len(pre_blocks) == len(k_neighbors) == len(reducers) == len(pos_blocks), \
            "Please check stage number consistent for pre_blocks, pos_blocks k_neighbors, reducers."
        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()
        last_channel = embed_dim
        anchor_points = self.points
        for i in range(len(pre_blocks)):
            out_channel = last_channel * expansions[i]
            pre_block_num = pre_blocks[i]
            pos_block_num = pos_blocks[i]
            kneighbor = k_neighbors[i]
            reduce = reducers[i]
            anchor_points = anchor_points // reduce

            # append local_grouper_list
            local_grouper = LocalGrouper(anchor_points, kneighbor, norm_augmented=norm_augmented,
                                         concat_anchor=concat_anchor)  # [b,g,k,d]
            self.local_grouper_list.append(local_grouper)
            # append pre_block_list
            temp_channel = last_channel * 2 if concat_anchor else last_channel
            pre_block_module = PreExtraction(temp_channel, out_channel, pre_block_num, heads=heads[i],
                                             head_dim=head_dims[i], activation=activation, similarity=smilarity,
                                             ffn_ratio=ffn_ratio)
            self.pre_blocks_list.append(pre_block_module)
            # append pos_block_list
            pos_block_module = PosExtraction(out_channel, pos_block_num, heads=heads[i], head_dim=head_dims[i],
                                             activation=activation, similarity=smilarity, ffn_ratio=ffn_ratio,
                                             norm_augmented=norm_augmented)
            self.pos_blocks_list.append(pos_block_module)

            last_channel = out_channel

        self.classifier = nn.Sequential(
            nn.Linear(last_channel, 512),
            nn.BatchNorm1d(512),
            get_activation(activation),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            get_activation(activation),
            nn.Dropout(0.5),
            nn.Linear(256, self.class_num)
        )

    def forward(self, x):  # [b,3,n]
        xyz = x.permute(0, 2, 1)  # [b,n,3]
        x = self.embedding(x)  # B,D,N
        for i in range(self.stages):
            # new_xyz[b,g,3], augmented_xyz [b,g,k,10], and new_fea[b,g,k,d]
            xyz, augmented_xyz, x = self.local_grouper_list[i](xyz, x.permute(0, 2, 1))  # [b,g,3][B,g,k,3][b,g,k,d]
            x = self.pre_blocks_list[i](x, augmented_xyz)  # [b,d,g]
            x = self.pos_blocks_list[i](x, xyz)  # [b,d,g] [b,d,3]

        x = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)
        x = self.classifier(x)
        return x


def pointsformer2A(num_classes=40, **kwargs) -> Pointsformer2:
    return Pointsformer2(points=1024, class_num=num_classes, embed_dim=32, activation='relu', smilarity='dot', ffn_ratio=0.125,
                 heads=[16, 16, 16, 16], head_dims=[64, 64, 64, 64], norm_augmented=True, concat_anchor=True,
                 expansions=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[2, 2, 2, 2], **kwargs)


def pointsformer2B(num_classes=40, **kwargs) -> Pointsformer2:
    return Pointsformer2(points=1024, class_num=num_classes, embed_dim=48, activation='relu', smilarity='dot', ffn_ratio=0.125,
                 heads=[16, 16, 16, 16], head_dims=[64, 64, 64, 64], norm_augmented=True, concat_anchor=True,
                 expansions=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[2, 2, 2, 2], **kwargs)

def pointsformer2C(num_classes=40, **kwargs) -> Pointsformer2:
    return Pointsformer2(points=1024, class_num=num_classes, embed_dim=64, activation='relu', smilarity='dot', ffn_ratio=0.125,
                 heads=[16, 16, 16, 16], head_dims=[64, 64, 64, 64], norm_augmented=True, concat_anchor=True,
                 expansions=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[2, 2, 2, 2], **kwargs)

def pointsformer2D(num_classes=40, **kwargs) -> Pointsformer2:
    return Pointsformer2(points=1024, class_num=num_classes, embed_dim=64, activation='gelu', smilarity='dot', ffn_ratio=0.125,
                 heads=[16, 16, 16, 16], head_dims=[64, 64, 64, 64], norm_augmented=True, concat_anchor=True,
                 expansions=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[2, 2, 2, 2], **kwargs)

def pointsformer2E(num_classes=40, **kwargs) -> Pointsformer2:
    return Pointsformer2(points=1024, class_num=num_classes, embed_dim=64, activation='relu', smilarity='dot', ffn_ratio=0.25,
                 heads=[16, 16, 16, 16], head_dims=[64, 64, 64, 64], norm_augmented=True, concat_anchor=True,
                 expansions=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[2, 2, 2, 2], **kwargs)

def pointsformer2F(num_classes=40, **kwargs) -> Pointsformer2:
    return Pointsformer2(points=1024, class_num=num_classes, embed_dim=64, activation='relu', smilarity='dot', ffn_ratio=0.125,
                 heads=[4, 4, 4, 4], head_dims=[64, 64, 64, 64], norm_augmented=True, concat_anchor=True,
                 expansions=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[2, 2, 2, 2], **kwargs)

def pointsformer2G(num_classes=40, **kwargs) -> Pointsformer2:
    return Pointsformer2(points=1024, class_num=num_classes, embed_dim=64, activation='relu', smilarity='dot', ffn_ratio=0.125,
                 heads=[8, 8, 8, 8], head_dims=[64, 64, 64, 64], norm_augmented=True, concat_anchor=True,
                 expansions=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[2, 2, 2, 2], **kwargs)

def pointsformer2H(num_classes=40, **kwargs) -> Pointsformer2:
    return Pointsformer2(points=1024, class_num=num_classes, embed_dim=64, activation='relu', smilarity='dot', ffn_ratio=0.125,
                 heads=[8, 8, 8, 8], head_dims=[32, 32, 32, 32], norm_augmented=True, concat_anchor=True,
                 expansions=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[2, 2, 2, 2], **kwargs)

def pointsformer2I(num_classes=40, **kwargs) -> Pointsformer2:
    return Pointsformer2(points=1024, class_num=num_classes, embed_dim=64, activation='relu', smilarity='dot', ffn_ratio=0.125,
                 heads=[8, 8, 8, 8], head_dims=[32, 32, 32, 32], norm_augmented=False, concat_anchor=True,
                 expansions=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[2, 2, 2, 2], **kwargs)

def pointsformer2J(num_classes=40, **kwargs) -> Pointsformer2:
    return Pointsformer2(points=1024, class_num=num_classes, embed_dim=32, activation='relu', smilarity='dot', ffn_ratio=0.125,
                 heads=[16, 16, 16, 16], head_dims=[64, 64, 64, 64], norm_augmented=True, concat_anchor=False,
                 expansions=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[2, 2, 2, 2], **kwargs)

def pointsformer2K(num_classes=40, **kwargs) -> Pointsformer2:
    return Pointsformer2(points=1024, class_num=num_classes, embed_dim=32, activation='relu', smilarity='dot', ffn_ratio=0.125,
                 heads=[16, 16, 16, 16], head_dims=[64, 64, 64, 64], norm_augmented=True, concat_anchor=True,
                 expansions=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2], **kwargs)

def pointsformer2L(num_classes=40, **kwargs) -> Pointsformer2:
    return Pointsformer2(points=1024, class_num=num_classes, embed_dim=32, activation='relu', smilarity='dot', ffn_ratio=0.125,
                 heads=[16, 16, 16, 16], head_dims=[64, 64, 64, 64], norm_augmented=True, concat_anchor=True,
                 expansions=[2, 2, 2, 2], pre_blocks=[2, 4, 6, 3], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2], **kwargs)


if __name__ == '__main__':
    """
    data = torch.rand(2,128,10)
    att = Attention(128)
    out = att(data)
    print(out.shape)



    batch, groups,neighbors,dim=2,512,32,16
    x = torch.rand(batch,groups,neighbors,dim)
    pre_extractor = PreExtraction(dim,3)
    out = pre_extractor(x)
    print(out.shape)

    x = torch.rand(batch, dim, groups)
    pos_extractor = PosExtraction(dim, 3)
    out = pos_extractor(x)
    print(out.shape)


    data = torch.rand(2, 3, 1024)
    print("===> testing model ...")
    model = Pointsformer2()
    out = model(data)
    print(out.shape)
    """

    data = torch.rand(2, 3, 1024)
    model = Pointsformer2()
    out = model(data)
    print(out.shape)
