import matplotlib.pyplot as plt
import numpy as np
import torch

dtype = torch.double
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

N = 64
D_in = 1
H = 100
D_out = 1

x = torch.rand(N, D_in, device=device, dtype=dtype)
y = torch.rand(N, D_out, device=device, dtype=dtype)

w1 = torch.rand(D_in, H, device=device, dtype=dtype, requires_grad=True)
b1 = torch.rand(H, device=device, dtype=dtype, requires_grad=True)

w2 = torch.rand(H, D_out, device=device, dtype=dtype, requires_grad=True)
b2 = torch.rand(D_out, device=device, dtype=dtype, requires_grad=True)

lr = 1e-4

for t in range(5000):
    y_pred = (x.mm(w1) + b1).clamp(0).mm(w2) + b2

    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    loss.backward()
    w1.data -= lr * w1.grad
    w2.data -= lr * w2.grad
    b1.data -= lr * b1.grad
    b2.data -= lr * b2.grad

    w1.grad.zero_()
    w2.grad.zero_()
    b1.grad.zero_()
    b2.grad.zero_()

plt.plot(x.cpu().numpy()[:, 0], y.cpu().numpy()[:, 0], 'x')
x = np.linspace(0, 1, 1000)
x_tensor = torch.from_numpy(x).to(device).view(1000, 1)
y_tensor = (x_tensor.mm(w1) + b1).clamp(0).mm(w2) + b2
y = y_tensor.detach().cpu().numpy()
plt.plot(x, y)
plt.show()
