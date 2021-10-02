"""
Embed PAConv into DGCNN
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from cuda_lib.functional import assign_score_withk as assemble_dgcnn

def knn(x, k):
    B, _, N = x.size()
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    _, idx = pairwise_distance.topk(k=k, dim=-1)  # (batch_size, num_points, k)

    return idx, pairwise_distance


def get_scorenet_input(x, idx, k):
    """(neighbor, neighbor-center)"""
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()

    neighbor = x.view(batch_size * num_points, -1)[idx, :]

    neighbor = neighbor.view(batch_size, num_points, k, num_dims)

    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    xyz = torch.cat((neighbor - x, neighbor), dim=3).permute(0, 3, 1, 2)  # b,6,n,k

    return xyz


def feat_trans_dgcnn(point_input, kernel, m):
    """transforming features using weight matrices"""
    # following get_graph_feature in DGCNN: torch.cat((neighbor - center, neighbor), dim=3)
    B, _, N = point_input.size()  # b, 2cin, n
    point_output = torch.matmul(point_input.permute(0, 2, 1).repeat(1, 1, 2), kernel).view(B, N, m, -1)  # b,n,m,cout
    center_output = torch.matmul(point_input.permute(0, 2, 1), kernel[:point_input.size(1)]).view(B, N, m, -1)  # b,n,m,cout
    return point_output, center_output


def feat_trans_pointnet(point_input, kernel, m):
    """transforming features using weight matrices"""
    # no feature concat, following PointNet
    B, _, N = point_input.size()  # b, cin, n
    point_output = torch.matmul(point_input.permute(0, 2, 1), kernel).view(B, N, m, -1)  # b,n,m,cout
    return point_output


class ScoreNet(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_unit=[16], last_bn=False):
        super(ScoreNet, self).__init__()
        self.hidden_unit = hidden_unit
        self.last_bn = last_bn
        self.mlp_convs_hidden = nn.ModuleList()
        self.mlp_bns_hidden = nn.ModuleList()

        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs_nohidden = nn.Conv2d(in_channel, out_channel, 1, bias=not last_bn)
            if self.last_bn:
                self.mlp_bns_nohidden = nn.BatchNorm2d(out_channel)

        else:
            self.mlp_convs_hidden.append(nn.Conv2d(in_channel, hidden_unit[0], 1, bias=False))  # from in_channel to first hidden
            self.mlp_bns_hidden.append(nn.BatchNorm2d(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):  # from 2nd hidden to next hidden to last hidden
                self.mlp_convs_hidden.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1, bias=False))
                self.mlp_bns_hidden.append(nn.BatchNorm2d(hidden_unit[i]))
            self.mlp_convs_hidden.append(nn.Conv2d(hidden_unit[-1], out_channel, 1, bias=not last_bn))  # from last hidden to out_channel
            self.mlp_bns_hidden.append(nn.BatchNorm2d(out_channel))

    def forward(self, xyz, calc_scores='softmax', bias=0):
        B, _, N, K = xyz.size()
        scores = xyz

        if self.hidden_unit is None or len(self.hidden_unit) == 0:
            if self.last_bn:
                scores = self.mlp_bns_nohidden(self.mlp_convs_nohidden(scores))
            else:
                scores = self.mlp_convs_nohidden(scores)
        else:
            for i, conv in enumerate(self.mlp_convs_hidden):
                if i == len(self.mlp_convs_hidden)-1:  # if the output layer, no ReLU
                    if self.last_bn:
                        bn = self.mlp_bns_hidden[i]
                        scores = bn(conv(scores))
                    else:
                        scores = conv(scores)
                else:
                    bn = self.mlp_bns_hidden[i]
                    scores = F.relu(bn(conv(scores)))

        if calc_scores == 'softmax':
            scores = F.softmax(scores, dim=1)+bias  # B*m*N*K, where bias may bring larger gradient
        elif calc_scores == 'sigmoid':
            scores = torch.sigmoid(scores)+bias  # B*m*N*K
        else:
            raise ValueError('Not Implemented!')

        scores = scores.permute(0, 2, 3, 1)  # B*N*K*m

        return scores


class PAConv(nn.Module):
    def __init__(self):
        super(PAConv, self).__init__()
        self.k = 20
        self.calc_scores = "softmax"

        self.m1, self.m2, self.m3, self.m4 = 8,8,8,8
        self.scorenet1 = ScoreNet(6, self.m1, hidden_unit=[16])
        self.scorenet2 = ScoreNet(6, self.m2, hidden_unit=[16])
        self.scorenet3 = ScoreNet(6, self.m3, hidden_unit=[16])
        self.scorenet4 = ScoreNet(6, self.m4, hidden_unit=[16])

        i1 = 3  # channel dim of input_1st
        o1 = i2 = 64  # channel dim of output_1st and input_2nd
        o2 = i3 = 64  # channel dim of output_2st and input_3rd
        o3 = i4 = 128  # channel dim of output_3rd and input_4th
        o4 = 256  # channel dim of output_4th

        tensor1 = nn.init.kaiming_normal_(torch.empty(self.m1, i1 * 2, o1), nonlinearity='relu') \
            .permute(1, 0, 2).contiguous().view(i1 * 2, self.m1 * o1)
        tensor2 = nn.init.kaiming_normal_(torch.empty(self.m2, i2 * 2, o2), nonlinearity='relu') \
            .permute(1, 0, 2).contiguous().view(i2 * 2, self.m2 * o2)
        tensor3 = nn.init.kaiming_normal_(torch.empty(self.m3, i3 * 2, o3), nonlinearity='relu') \
            .permute(1, 0, 2).contiguous().view(i3 * 2, self.m3 * o3)
        tensor4 = nn.init.kaiming_normal_(torch.empty(self.m4, i4 * 2, o4), nonlinearity='relu') \
            .permute(1, 0, 2).contiguous().view(i4 * 2, self.m4 * o4)

        # convolutional weight matrices in Weight Bank:
        self.matrice1 = nn.Parameter(tensor1, requires_grad=True)
        self.matrice2 = nn.Parameter(tensor2, requires_grad=True)
        self.matrice3 = nn.Parameter(tensor3, requires_grad=True)
        self.matrice4 = nn.Parameter(tensor4, requires_grad=True)

        self.bn1 = nn.BatchNorm1d(o1, momentum=0.1)
        self.bn2 = nn.BatchNorm1d(o2, momentum=0.1)
        self.bn3 = nn.BatchNorm1d(o3, momentum=0.1)
        self.bn4 = nn.BatchNorm1d(o4, momentum=0.1)
        self.bn5 = nn.BatchNorm1d(1024, momentum=0.1)
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   self.bn5)

        self.linear1 = nn.Linear(2048, 512, bias=False)
        self.bn11 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn22 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, 40)

    def forward(self, x):
        B, C, N = x.size()
        idx, _ = knn(x, k=self.k)  # different with DGCNN, the knn search is only in 3D space
        xyz = get_scorenet_input(x, idx=idx, k=self.k)  # ScoreNet input: 3D coord difference concat with coord: b,6,n,k

        ##################
        # replace all the DGCNN-EdgeConv with PAConv:
        """CUDA implementation of PAConv: (presented in the supplementary material of the paper)"""
        """feature transformation:"""
        point1, center1 = feat_trans_dgcnn(point_input=x, kernel=self.matrice1, m=self.m1)  # b,n,m1,o1
        score1 = self.scorenet1(xyz, calc_scores=self.calc_scores, bias=0.5)
        """assemble with scores:"""
        point1 = assemble_dgcnn(score=score1, point_input=point1, center_input=center1, knn_idx=idx, aggregate='sum')  # b,o1,n
        point1 = F.relu(self.bn1(point1))

        point2, center2 = feat_trans_dgcnn(point_input=point1, kernel=self.matrice2, m=self.m2)
        score2 = self.scorenet2(xyz, calc_scores=self.calc_scores, bias=0.5)
        point2 = assemble_dgcnn(score=score2, point_input=point2, center_input=center2, knn_idx=idx, aggregate='sum')
        point2 = F.relu(self.bn2(point2))

        point3, center3 = feat_trans_dgcnn(point_input=point2, kernel=self.matrice3, m=self.m3)
        score3 = self.scorenet3(xyz, calc_scores=self.calc_scores, bias=0.5)
        point3 = assemble_dgcnn(score=score3, point_input=point3, center_input=center3, knn_idx=idx, aggregate='sum')
        point3 = F.relu(self.bn3(point3))

        point4, center4 = feat_trans_dgcnn(point_input=point3, kernel=self.matrice4, m=self.m4)
        score4 = self.scorenet4(xyz, calc_scores=self.calc_scores, bias=0.5)
        point4 = assemble_dgcnn(score=score4, point_input=point4, center_input=center4, knn_idx=idx, aggregate='sum')
        point4 = F.relu(self.bn4(point4))
        ##################

        point = torch.cat((point1, point2, point3, point4), dim=1)
        point = F.relu(self.conv5(point))
        point11 = F.adaptive_max_pool1d(point, 1).view(B, -1)
        point22 = F.adaptive_avg_pool1d(point, 1).view(B, -1)
        point = torch.cat((point11, point22), 1)

        point = F.relu(self.bn11(self.linear1(point)))
        point = self.dp1(point)
        point = F.relu(self.bn22(self.linear2(point)))
        point = self.dp2(point)
        point = self.linear3(point)

        return point
