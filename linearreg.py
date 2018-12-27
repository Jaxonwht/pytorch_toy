import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

gpu = torch.device("cuda")

a = 10
D_in = 1
H = 20
D_out = 1
epochs = 10000
x = torch.rand(100, 1, device=gpu) * a
y = x.pow(2) + torch.rand(100, 1, device=gpu)


class LinearModel(nn.Module):
    def __init__(self, D_in, H, D_out):
        super().__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)

    def forward(self, x):
        return self.linear2(torch.sigmoid(self.linear1(x)))


loss_fn = nn.MSELoss()
model = LinearModel(D_in, H, D_out).cuda(gpu)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
for t in range(epochs):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    print(t, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.eval()
train_x = np.random.rand()
