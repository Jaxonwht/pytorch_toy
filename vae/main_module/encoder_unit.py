import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pad_sequence


class Encoder(nn.Module):
    def __init__(self, embed, hidden, num_layers=1, bidirectional=True, max_seq_len=50):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.mu = nn.GRU(input_size=embed, hidden_size=hidden, num_layers=num_layers,
                         bidirectional=bidirectional, batch_first=True)
        self.logvar = nn.GRU(input_size=embed, hidden_size=hidden, num_layers=num_layers, bidirectional=bidirectional,
                             batch_first=True)

    def forward(self, x, lengths):
        '''
        :param input: list of tensors of variable lengths, len(list) = batch_size, each entry in one element of the batch is [variable_seq_len, embed_dim]
        :param lengths: [batch_size]
        :return: (out, hidden, lengths, kl_loss), out is [batch, padded_seq_len, 2 x encoder_hidden_dim], hidden is [batch, 2 x encoder_hidden_dim], lengths is a list [batch_size], kl_loss is scalar
        '''
        x = pad_sequence(input, batch_first=True)
        x = pack_padded_sequence(x, lengths=lengths, batch_first=True)
        mu_out, mu_hidden = self.mu(x)
        logvar_out, logvar_hidden = self.logvar(x)
        # out is a padded seq
        # hidden = [2 x encoder_hidden_dim, batch]
        # hidden can be separated into part = hidden.view(num_layers, num_directions, batch, hidden_size)
        mu_out_list, _ = pad_packed_sequence(mu_out, batch_first=True, total_length=self.max_seq_len)
        logvar_out_list, _ = pad_packed_sequence(logvar_out, batch_first=True, total_length=self.max_seq_len)
        # both out lists = [batch, padded_seq_len, 2 x encoder_hidden_dim]
        out = torch.zeros(len(lengths), self.max_seq_len, 2 * self.mu.hidden_size)
        kl_loss = torch.zeros(1)
        for batch in range(len(lengths)):
            out[batch, :lengths[batch].item()] = self.sample(mu_out_list[batch, :lengths[batch].item()],
                                                             logvar_out_list[batch, :lengths[batch].item()])
            kl_loss = kl_loss.add(
                self.kl_convergence_loss(mu_out_list[batch, :lengths[batch].item()],
                                         logvar_out_list[batch, :lengths[batch].item()]))
        kl_loss = kl_loss.add(self.kl_convergence_loss(mu_hidden, logvar_hidden)).div(len(lengths))
        hidden = self.sample(mu_hidden, logvar_hidden)
        # out = [batch, max_seq_len, 2 x encoder_hidden_dim]
        # hidden = [2 x encoder_hidden_dim, batch]
        return out, hidden, kl_loss

    def sample(self, mu, logvar):
        '''
        :param mu: [embedding_dim]
        :param logvar: [embedding_dim]
        :return: [embedding_dim]
        '''
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        return mu + torch.mul(eps, std)

    def kl_convergence_loss(self, mu, logvar):
        return 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1 - logvar)


if __name__ == "__main__":
<<<<<<< HEAD
    input = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
    embedding = nn.Embedding(6, 2)
    input = [embedding(x) for x in input]
    lengths = torch.tensor([3, 2])
    model = Encoder(2, 10)
    out, hidden, lengths, kl_loss = model(input, lengths)
=======
    input = [torch.randn(3, 10), torch.randn(2,10)]
    lengths = torch.tensor([3, 2])
    model = Encoder(embed=10, hidden=2)
    out, hidden, kl_loss = model(input, lengths)
>>>>>>> decoder
    print(out.size())
    print(hidden.size())
    print(kl_loss)
    print(lengths)
