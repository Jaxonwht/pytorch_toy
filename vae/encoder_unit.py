import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, N, V, E, H, L, B):
        super().__init__()
        self.embedding = nn.Embedding(V, E)
        self.lstm = nn.LSTM(E, H, num_layers=L, bidirectional=B)

    def init_embedding(self, pretrained):
        self.embedding.weight.data = pretrained.data
