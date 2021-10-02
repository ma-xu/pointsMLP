from torchsummary import summary
from CurveNet import CurveNet as net  #2,142,672
# from modelelite3 import modelelite3X10 as net
# from GBNet import GBNet as net  # 8,798,048
# from GDANet import GDANET as net  # 939,176
# from GBNet import DGCNN as net #1,812,648
# from pointConv import PointConv as net # 19,569,657
model = net().cuda()
print(summary(model, (3,1024)))
