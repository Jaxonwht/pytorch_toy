import torch
import torch.nn as nn

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

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
