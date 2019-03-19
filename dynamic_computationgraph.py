import random

import torch
import torch.nn as nn
import torch.optim as optim


class DynamicNet(nn.Module):
    def __init__(self, D_in, H, D_out):
        super().__init__()
        self.input = nn.Linear(D_in, H)
        self.activation = nn.Sigmoid()
        self.middle = nn.Linear(H, H)
        self.output = nn.Linear(H, D_out)

    def forward(self, x):
        h_i = self.activation(self.input(x))
        for i in range(random.randint(0, 3)):
            h_i = self.activation(self.middle(h_i))
        return self.output(h_i)


gpu = torch.device("cuda")
N = 64
D_in = 1000
H = 100
D_out = 10

x = torch.rand(N, D_in, device=gpu)
y = torch.rand(N, D_out, device=gpu)

learning_rate = 1e-4
epochs = 5000

dynamic_model = DynamicNet(D_in, H, D_out).cuda(gpu)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(dynamic_model.parameters())

for i in range(epochs):
    y_pred = dynamic_model(x)
    loss = loss_fn(y_pred, y)
    print(i, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
