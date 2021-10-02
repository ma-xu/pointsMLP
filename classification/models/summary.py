from torchsummary import summary
from CurveNet import CurveNet as net
model = net().cuda()
print(summary(model, (3,1024)))
