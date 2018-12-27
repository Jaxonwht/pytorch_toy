import torch.nn as nn
import torch

if torch.cuda.is_available():
    gpu = torch.device("gpu")
else:
    gpu = torch.device("cpu")

BATCH_SIZE = 32
EMBEDDING_SIZE = 500
HIDDEN_SIZE = 300
LEARNING_RATE = 1e-3
EPOCHS = 300
SEQ_LENGTH = 100
NUM_OF_LAYERS = 1
BI_DIRECTIONAL = True


class Encoder(nn.Module):
    def __init__(self, N, V, E, H, L, B):
        super().__init__()
        self.embedding = nn.Embedding(V, E)
        self.lstm = nn.LSTM(E, H, num_layers=L, bidirectional=B)


    def init_embedding(self, pretrained):
        self.embedding.weight.data = pretrained.data

