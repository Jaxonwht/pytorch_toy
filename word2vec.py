import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import EmailDataset
from torch.utils.data import DataLoader

file_path = "data/emails.train"
gpu = torch.device("cuda")
context_size = 2
batch_size = 32
hidden_size = 300
learning_rate = 1e-4
epochs = 10

email_data = EmailDataset(file_path, context_size)
vocab_size = email_data.getNumberOfToken()
my_data_loader = DataLoader(email_data, shuffle=True, batch_size=batch_size)

class Word2Vec(nn.Module):
    def __init__(self, D, H):
        super().__init__()
        self.input = nn.Linear(D, H)
        self.output = nn.Linear(H, D)

    def forward(self, *input):
        return self.output(self.input(*input))

model = Word2Vec(vocab_size, hidden_size).cuda(gpu)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for e in range(epochs):
    for x, y in enumerate(my_data_loader):
        input_vector = torch.zeros(y[0].size()[0], vocab_size, device=gpu)
        for i in range(y[0].size()[0]):
            input_vector[i][y[0][i]] = 1
        target_pred = model(input_vector)
        target_vector = torch.tensor(y[1], device=gpu)
        loss = loss_fn(target_pred, target_vector)
        if ((x + 1) % 1000 == 0):
            print("Epoch: {0}, Batch: {1}, loss: {2}".format(e, x, loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()