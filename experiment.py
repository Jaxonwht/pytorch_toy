import torch

x = torch.rand(5, 1)

y = x.expand(5, 6)

print(y)
y[1] = torch.rand(6, 1)

print(y)