import torch
import torch.nn as nn

N = 64
D_in = 1000
H = 100
D_out = 10
gpu = torch.device("cuda")

x = torch.rand(N, D_in, device=gpu)
y = torch.rand(N, D_out, device=gpu)

model = nn.Sequential(
    nn.Linear(D_in, H),
    nn.ReLU(),
    nn.Linear(H, D_out),
).cuda(gpu)

loss_fn = nn.MSELoss(reduction="sum")

lr = 1e-4
epochs = 4000

for t in range(epochs):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    print(t, loss.item())
    model.zero_grad()
    loss.backward()
    with torch.no_grad():
        for param in model.parameters():
            param -= lr * param.grad
