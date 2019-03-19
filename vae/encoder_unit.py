import torch
import torch.nn as nn

from vae.data_loader import VAEData


class Encoder(nn.Module):
    def __init__(self, embed, hidden, num_layers, bidirectional):
        super().__init__()
        self.gru = nn.GRU(input_size=embed, hidden_size=hidden, num_layers=num_layers,
                          bidirectional=bidirectional)

    def forward(self, x):
        _, hidden = self.gru(x)
        return hidden


if __name__ == "__main__":
    data = VAEData("../data/democratic_only.train.en")
    print(len(data))
    print(data[0])
    print(data[1])
    rnn = nn.GRU(10, 20, 2)
    input = torch.randn(5, 3, 10)
    h0 = torch.randn(2, 3, 20)
    output, hn = rnn(input, h0)
