import torch
import torch.nn as nn
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import GRU
from torch.nn.modules.sparse import Embedding
from torch.nn.utils.rnn import pack_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.optim import Adam

from vae.data_loader.data_loader import VAEData
from vae.main_module.encoder_unit import Encoder


def convert_batch_to_sorted_list_pack(x):
    x.sort(key=lambda seq: len(seq), reverse=True)
    return pack_sequence(x)


class VAE(nn.Module):
    def __init__(self, embed, vocabulary, hidden, embedding_weight, num_layers, bidirectional, middle):
        super().__init__()
        self.embedding = Embedding(vocabulary, embed)
        self.encoder_mu = Encoder(embed=embed, hidden=hidden, num_layers=num_layers, bidirectional=bidirectional)
        self.encoder_logvar = Encoder(embed=embed, hidden=hidden, num_layers=num_layers, bidirectional=bidirectional)
        self.decoder = GRU(input_size=embed, hidden_size=hidden, num_layers=num_layers, bidirectional=bidirectional)
        self.fc1 = Linear(in_features=hidden * 2, out_features=middle)
        self.relu = nn.LeakyReLU(0.1)
        self.fc2 = Linear(in_features=middle, out_features=vocabulary)

    def sample(self):
        std = torch.exp(0.5 * self.hidden_logvar)
        eps = torch.rand_like(std)
        return self.hidden_mu + torch.mul(eps, std)

    def forward(self, x):
        x = [self.embedding(seq) for seq in x]
        pack = convert_batch_to_sorted_list_pack(x)
        self.hidden_mu = self.encoder_mu(pack)
        self.hidden_logvar = self.encoder_logvar(pack)
        sample_context = self.sample()
        out, _ = self.decoder(pack, sample_context)
        output, lengths = pad_packed_sequence(out, batch_first=True)
        output = [output[i, : lengths[i]] for i in range(len(lengths))]
        output = [self.fc2(self.relu(self.fc1(hidden_vector))) for hidden_vector in output]
        return output, x

    def kl_convergence(self):
        return 0.5 * torch.sum(self.hidden_mu.pow(2) + self.hidden_logvar.exp() - 1 - self.hidden_mu)

    def train(self, learning_rate, epochs, batch_size, dataset, model_out, optim_out):
        optimizer = Adam(self.parameters(), lr=learning_rate)
        for e in range(epochs):
            for batch_num in range(len(dataset) // batch_size):
                list_of_sequences = dataset[batch_num * batch_size: batch_num * batch_size + batch_size]
                output, input = self.forward(list_of_sequences)
                loss_fn = nn.CrossEntropyLoss(reduction="sum")
                loss = torch.zeros(1)
                for i in range(batch_size):
                    loss.add_(loss_fn(output[:-1], input[1:]))
                loss = loss.div(batch_size)
                loss = loss.add(self.kl_convergence())
                if batch_num % 1000 == 0:
                    print("Epoch: {}, Batch: {}, loss: {}".format(e, batch_num, loss))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                torch.save(self.state_dict(), model_out)
                torch.save(optimizer.state_dict(), optim_out)
        torch.save(self.state_dict(), model_out)
        torch.save(optimizer.state_dict(), optim_out)


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
    MIDDLE = 100
    bidirectional = True

    training = "../../data/democratic_only.dev.en"
    training_dataset = VAEData(training)
    model = VAE(EMBEDDING_SIZE, training_dataset.get_vocab_size(), HIDDEN_SIZE,
                torch.load("../../word2vec/model/model_state_dict.pt")["embed.weight"], NUM_OF_LAYERS,
                bidirectional=True, middle=MIDDLE).cuda()
    input = [training_dataset[i].to(my_device) for i in range(2)]
    output, x = model(input)
