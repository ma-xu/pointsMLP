import torch
import torch.nn as nn


# permutation-invariant MLP
class PIConv(nn.Module):
    def __init__(self, channels, neighbors, reduction=16, bias=False, activation='relu', scale=True):
        super(PIConv, self).__init__()
        self.reduction = reduction
        self.scale = scale
        self.mid_channel = channels
        self.before_affine_alpha = nn.Parameter(torch.ones(1, channels, 1))
        self.before_affine_beta = nn.Parameter(torch.zeros(1, channels, 1))
        self.after_affine_alpha = nn.Parameter(torch.ones(1, channels, 1))
        self.after_affine_beta = nn.Parameter(torch.zeros(1, channels, 1))

        if self.reduction > 1:
            self.mid_channel = channels // reduction
            self.down = nn.Conv1d(channels, self.mid_channel, 1, bias=bias)
            self.up = nn.Conv1d(self.mid_channel, channels, 1, bias=bias)
            torch.nn.init.xavier_normal_(self.down.weight.data)
            torch.nn.init.xavier_normal_(self.up.weight.data)
        self.per_inv = nn.Conv1d(self.mid_channel, self.mid_channel * neighbors, 1, bias=bias)
        torch.nn.init.xavier_normal_(self.per_inv.weight.data)
        if activation.lower() == 'gelu':
            self.act = nn.GELU()
        elif activation.lower() == 'leakyrelu':
            self.act = nn.LeakyReLU(inplace=True)
        else:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):  # x: [b, c, k]

        x = x * self.before_affine_alpha + self.before_affine_beta
        if self.reduction > 1:
            x = self.act(self.down(x))
        b, c, k = x.shape
        x = self.act(self.per_inv(x)).reshape([b, c, k, k]).sum(dim=-1, keepdim=False)
        if self.scale:
            x /= k
        if self.reduction > 1:
            x = self.act(self.up(x))
        x = x * self.after_affine_alpha + self.after_affine_beta
        return x



class PILinear(nn.Module):
    def __init__(self, channels, neighbors, reduction=16, bias=False, activation='relu', scale=True):
        super(PILinear, self).__init__()
        self.reduction = reduction
        self.scale = scale
        self.mid_channel = channels
        self.before_affine_alpha = nn.Parameter(torch.ones(1, 1, channels))
        self.before_affine_beta = nn.Parameter(torch.zeros(1, 1, channels))
        self.after_affine_alpha = nn.Parameter(torch.ones(1, 1, channels))
        self.after_affine_beta = nn.Parameter(torch.zeros(1, 1, channels))

        if self.reduction > 1:
            self.mid_channel = channels // reduction
            self.down = nn.Linear(channels, self.mid_channel, bias=bias)
            self.up = nn.Linear(self.mid_channel, channels, bias=bias)
            torch.nn.init.xavier_normal_(self.down.weight.data)
            torch.nn.init.xavier_normal_(self.up.weight.data)
        self.per_inv = nn.Linear(self.mid_channel, self.mid_channel * neighbors, bias=bias)
        torch.nn.init.xavier_normal_(self.per_inv.weight.data)
        if activation.lower() == 'gelu':
            self.act = nn.GELU()
        elif activation.lower() == 'leakyrelu':
            self.act = nn.LeakyReLU(inplace=True)
        else:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):  # x: [b,k,c]

        x = x * self.before_affine_alpha + self.before_affine_beta
        if self.reduction > 1:
            x = self.act(self.down(x))
        b, k, c = x.shape
        x = self.act(self.per_inv(x)).reshape([b, k, c, k]).sum(dim=1, keepdim=False).permute(0, 2, 1)  # to [b,k,c]
        if self.scale:
            x /= k
        if self.reduction > 1:
            x = self.act(self.up(x))
        x = x * self.after_affine_alpha + self.after_affine_beta
        return x

if __name__ == '__main__':
    data = torch.rand(2, 128, 10)
    piconv = PIConv(channels=128, neighbors=10, reduction=1, bias=False, activation='leakyrelu')
    out = piconv(data)
    print(f"piconv: {out.shape}")

    data = torch.rand(2, 10, 128)
    pilinear = PILinear(channels=128, neighbors=10, reduction=2, bias=False, activation='leakyrelu')
    out = pilinear(data)
    print(f"pilinear: {out.shape}")
