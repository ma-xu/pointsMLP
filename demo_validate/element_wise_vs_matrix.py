import time
import torch
iterations =100
a = torch.rand(1000,1000)
b = torch.rand(1000,1000)

# current = time.time()
# for _ in range(iterations):
#     out = a*b
# print(f"The runnning time of element-wise multiplication is: {time.time()-current}")
#
# current = time.time()
# for _ in range(iterations):
#     out = torch.matmul(a,b)
# print(f"The runnning time of matrix multiplication is: {time.time()-current}")

a = torch.rand(32,512,64,64)
iterations=20
# heat up
for _ in range(iterations):
    out = torch.logsumexp(a,dim=-1)

current = time.time()
for _ in range(iterations):
    out = torch.logsumexp(a,dim=-1)
print(f"The CPU runnning time of logsumexp is: {time.time()-current}")

current = time.time()
for _ in range(iterations):
    _,out = torch.amax(a,dim=-1)
print(f"The CPU runnning time of max is: {time.time()-current}")

current = time.time()
for _ in range(iterations):
    out = torch.mean(a,dim=-1)
print(f"The CPU runnning time of mean is: {time.time()-current}")

current = time.time()
for _ in range(iterations):
    out = torch.norm(a,dim=-1)
print(f"The CPU runnning time of norm is: {time.time()-current}")



a = torch.rand(32,512,64,64)
a = a.to("cuda")
iterations=20
# heat up
for _ in range(iterations):
    out = torch.logsumexp(a,dim=-1)

current = time.time()
for _ in range(iterations):
    out = torch.logsumexp(a,dim=-1)
print(f"The GPU runnning time of logsumexp is: {time.time()-current}")

current = time.time()
for _ in range(iterations):
    out = torch.amax(a,dim=-1)
print(f"The GPU runnning time of max is: {time.time()-current}")

current = time.time()
for _ in range(iterations):
    out = torch.mean(a,dim=-1)
print(f"The GPU runnning time of mean is: {time.time()-current}")

current = time.time()
for _ in range(iterations):
    out = torch.norm(a,dim=-1)
print(f"The GPU runnning time of norm is: {time.time()-current}")

