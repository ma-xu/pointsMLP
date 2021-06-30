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
    out = torch.amax(a,dim=-1)
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

"""
The CPU runnning time of logsumexp is: 5.057518243789673
The CPU runnning time of max is: 0.09893107414245605
The CPU runnning time of mean is: 0.1019439697265625
The CPU runnning time of norm is: 3.853361129760742
The GPU runnning time of logsumexp is: 0.0012063980102539062
The GPU runnning time of max is: 0.00022935867309570312
The GPU runnning time of mean is: 0.00020766258239746094
The GPU runnning time of norm is: 0.0006072521209716797
"""
