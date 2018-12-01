import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import EmailDataset
from torch.utils.data import DataLoader
import torch.nn.init as init

file_path = "data/emails.train"
out_model_state_dict = "model/model_state_dict.pt"
out_optim_state_dict = "model/optim_state_dict.pt"
gpu = torch.device("cuda")
context_size = 2
batch_size = 32
hidden_size = 300
learning_rate = 1e-4
epochs = 50

email_data = EmailDataset(file_path, context_size)
vocab_size = email_data.getNumberOfToken()
my_data_loader = DataLoader(email_data, shuffle=True, batch_size=batch_size)

class Word2Vec(nn.Module):
    def __init__(self, D, H):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(D, H))
        self.bias = nn.Parameter(torch.zeros(H))
        init.xavier_uniform_(self.weight.data)
        init.constant_(self.bias.data, 0)
        self.output = nn.Linear(H, D)

    def forward(self, x):
        return self.output(self.weight[x] + self.bias)

model = Word2Vec(vocab_size, hidden_size).cuda(gpu)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for e in range(epochs):
   for x, y in enumerate(my_data_loader):
       input_vector = torch.tensor(y[0], device=gpu)
       target_pred = model(input_vector)
       target_vector = torch.tensor(y[1], device=gpu)
       loss = loss_fn(target_pred, target_vector)
       if ((x + 1) % 1000 == 0):
	       print("Epoch: {0}, Batch: {1}, loss: {2}".format(e, x, loss.item()))
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()

torch.save(model.state_dict(), out_model_state_dict)
torch.save(optimizer.state_dict(), out_optim_state_dict)
