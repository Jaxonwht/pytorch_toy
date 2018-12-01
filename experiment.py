import torch

x = torch.rand(5, 1)

z = torch.Tensor(4, 5)
print(z.size()[1])

y = x.expand(5, 6)


print(y)