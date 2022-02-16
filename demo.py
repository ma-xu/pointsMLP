import torch

x = torch.randn(10, 1)
w1 = torch.randn(1, 10, requires_grad=True)
w2 = torch.randn(1, 10, requires_grad=True)
out1 = torch.matmul(w1, x)
out2 = torch.matmul(w2, x).floor()
loss1 = (out1 - torch.randn(1, 1))**2
loss1.backward()
loss2 = (out2 - torch.randn(1, 1))**2
loss2.backward()
print(w1.grad)
print(w2.grad)
###########output:
# tensor([[ 0.2130,  1.6570,  0.8615,  0.7979, -0.1907, -0.2031, -0.0377,  0.9172,
#          -0.0924,  1.8134]])
# tensor([[0., 0., 0., 0., -0., -0., -0., 0., -0., 0.]])
# Process finished with exit code 0
