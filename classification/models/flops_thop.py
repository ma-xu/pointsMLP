import torch
from thop import profile
from pointsformer1 import pointsformer1H as net
model = net()
input = torch.randn(1, 3, 1024)
macs, params = profile(model, inputs=(input, ))
print(f"macs: {macs}")
print(f"params: {params}")
