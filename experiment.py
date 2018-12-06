import torch

x = torch.rand(1, 2, 3)

y = torch.rand(3, 4)

print(x[:, 0, :].size())

