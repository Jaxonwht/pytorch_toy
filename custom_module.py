import torch
import torch.nn as nn
import torch.optim as optim


class TwoLayer(nn.Module):
    def __init__(self, D_in, H, D_out):
        super().__init__()
        self.layer1 = nn.Linear(D_in, H)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(H, D_out)

    def forward(self, x):
        return self.layer2(self.activation(self.layer1(x)))


gpu = torch.device("cuda")
N = 64
D_in = 1000
H = 100
D_out = 10
epochs = 5000

x = torch.rand(N, D_in, device=gpu)
y = torch.rand(N, D_out, device=gpu)

model = TwoLayer(D_in, H, D_out).cuda(gpu)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

for i in range(epochs):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    print(i, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
