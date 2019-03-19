import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.modules.sparse import Embedding
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import GRU
from torch.nn.utils.rnn import pack_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from vae.data_loader import VAEData
from vae.encoder_unit import Encoder


def convert_batch_to_sorted_list_pack(x):
    x.sort(key=lambda seq: len(seq), reverse=True)
    return pack_sequence(x)


class VAE(nn.Module):
    def __init__(self, embed, vocabulary, hidden, embedding_weight, num_layers, bidirectional):
        super().__init__()
        self.embedding = Embedding.from_pretrained(embeddings=embedding_weight)
        print(embedding_weight.size())
        self.encoder_mu = Encoder(embed=embed, hidden=hidden, num_layers=num_layers, bidirectional=bidirectional)
        self.encoder_var = Encoder(embed=embed, hidden=hidden, num_layers=num_layers, bidirectional=bidirectional)
        self.decoder = GRU(input_size=embed, hidden_size=hidden, num_layers=num_layers, bidirectional=bidirectional)
        self.fc = Linear(in_features=hidden, out_features=vocabulary)

    def sample(self, mu, var):
        normal = torch.normal(mean=0, std=1)
        return mu + var * normal

    def forward(self, x):
        x = [self.embedding(seq) for seq in x]
        pack = convert_batch_to_sorted_list_pack(x)
        hidden_mu = self.encoder_mu(x)
        hidden_var = self.encoder_var(x)
        sample_context = self.sample(hidden_mu, hidden_var)
        out, _ = self.decoder(pack, sample_context)
        output, lengths = pad_packed_sequence(out, batch_first=True)
        output = [output[i, : lengths[i]] for i in len(lengths)]
        return output, x

    def train(self, learning_rate, epochs, batch_size, dataset):
        optimizer = Adam(self.parameters(), lr=learning_rate)
        for e in range(epochs):
            for batch_num in range(len(dataset) // batch_size):
                list_of_sequences = dataset[batch_num * batch_size: batch_num * batch_size + batch_size]


if __name__ == "__main__":
    if torch.cuda.is_available():
        my_device = torch.device("cuda")
    else:
        my_device = torch.device("cpu")
    BATCH_SIZE = 32
    HIDDEN_SIZE = 200
    LEARNING_RATE = 1e-3
    EPOCHS = 300
    NUM_OF_LAYERS = 1
    EMBEDDING_SIZE = 500
    bidirectional = True

    training = "../data/democratic_only.dev.en"
    training_dataset = VAEData(training)
    print(training_dataset.get_vocab_size())
    model = VAE(EMBEDDING_SIZE, training_dataset.get_vocab_size(), HIDDEN_SIZE, torch.load("../word2vec/model/model_state_dict.pt")["embed.weight"], NUM_OF_LAYERS, bidirectional=True).cuda()
    input = [training_dataset[i].to(my_device) for i in range(4)]
    print(input)
    print(model(input))
