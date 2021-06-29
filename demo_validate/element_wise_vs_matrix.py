import time
import torch
iterations =100
a = torch.rand(1000,1000)
b = torch.rand(1000,1000)
current = time.time()
for _ in range(iterations):
    out = a*b
print(f"The runnning time of element-wise multiplication is: {time.time()-current}")

current = time.time()
for _ in range(iterations):
    out = torch.matmul(a,b)
print(f"The runnning time of matrix multiplication is: {time.time()-current}")

