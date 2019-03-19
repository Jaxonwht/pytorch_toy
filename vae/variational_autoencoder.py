import torch
import torch.nn as nn
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import GRU
from torch.nn.utils.rnn import pack_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from vae.data_loader import VAEData
from vae.encoder_unit import Encoder


def convert_batch_to_sorted_list_pack(x):
    list_of_sequences = [x[i] for i in range(len(x))]
    list_of_sequences.sort(key=lambda seq: len(seq), reverse=True)
    return pack_sequence(list_of_sequences)


class VAE(nn.Module):
    def __init__(self, embed, vocabulary, hidden):
        super().__init__()
        self.encoder_mu = Encoder(embed=embed, hidden=hidden, num_layers=NUM_OF_LAYERS, bidirectional=BI_DIRECTIONAL)
        self.encoder_var = Encoder(embed=embed, hidden=hidden, num_layers=NUM_OF_LAYERS, bidirectional=BI_DIRECTIONAL)
        self.decoder = GRU(input_size=embed, hidden_size=hidden, num_layers=NUM_OF_LAYERS, bidirectional=BI_DIRECTIONAL)
        self.fc = Linear(in_features=hidden, out_features=vocabulary)

    def sample(self, mu, var):
        normal = torch.normal(mean=0, std=1)
        return mu + var * normal

    def forward(self, x):
        pack = convert_batch_to_sorted_list_pack(x)
        hidden_mu = self.encoder_mu(x)
        hidden_var = self.encoder_var(x)
        sample_context = self.sample(hidden_mu, hidden_var)
        out, _ = self.decoder(pack, sample_context)
        unpack, lengths = pad_packed_sequence(out, batch_first=True)
        return unpack, lengths


if __name__ == "__main__":
    if torch.cuda.is_available():
        my_device = torch.device("gpu")
    else:
        my_device = torch.device("cpu")
    BATCH_SIZE = 32
    HIDDEN_SIZE = 200
    LEARNING_RATE = 1e-3
    EPOCHS = 300
    NUM_OF_LAYERS = 1
    BI_DIRECTIONAL = True

    training = "../data/democratic_only.train.en"
    training_dataset = VAEData(training)
    print(training_dataset.get_vocab_size())
    print(len(training_dataset))
