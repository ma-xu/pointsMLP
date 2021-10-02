from torchsummary import summary
from CurveNet import CurveNet as net  #2,142,672
# from modelelite3 import modelelite3X10 as net
from GBNet import GBNet as net
model = net().cuda()
print(summary(model, (3,1024)))
