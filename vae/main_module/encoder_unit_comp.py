import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self, hidden, embed):
        super().__init__()
        self.mu = nn.GRU(input_size=embed, hidden_size=hidden, num_layers=1,
                         bidirectional=True, batch_first=True)
        self.logvar = nn.GRU(input_size=embed, hidden_size=hidden, num_layers=1, bidirectional=True,
                             batch_first=True)

    def forward(self, x, lengths, variation):
        '''
        :param input: list of variable lengths of tensors, len(list) = batch_size, each entry in one element of the batch is [variable_seq_len]
        :param lengths: list, len(list) = batch_size
        :return: (out, hidden, kl_loss), out is [batch, padded_seq_len, 2 x encoder_hidden_dim], hidden is [batch, 2 x encoder_hidden_dim], kl_loss is scalar
        '''
        if variation:
            mu_out, _ = self.mu(x)
            logvar_out, _ = self.logvar(x)
            # out is a padded seq
            # hidden = [2, batch, encoder_hidden_dim]
            # hidden can be separated into part = hidden.view(num_layers, num_directions, batch, hidden_size)
            mu_out_list, _ = pad_packed_sequence(mu_out, batch_first=True)
            logvar_out_list, _ = pad_packed_sequence(logvar_out, batch_first=True)
            # both out lists = [batch, padded_seq_len, 2 x encoder_hidden_dim]
            out = Variable(torch.zeros(len(lengths), lengths[0], 2 * self.mu.hidden_size).cuda())
            kl_loss = Variable(torch.zeros(1).cuda())
            for batch in range(len(lengths)):
                out[batch, :lengths[batch]] = self.sample(mu_out_list[batch, :lengths[batch]],
                                                          logvar_out_list[batch, :lengths[batch]])
                kl_loss = kl_loss.add(
                    self.kl_convergence_loss(mu_out_list[batch, :lengths[batch]],
                                             logvar_out_list[batch, :lengths[batch]]))
            kl_loss = kl_loss.div(len(lengths))
            hidden = out[:, -1, :]
            # out = [batch, max_seq_len, 2 x encoder_hidden_dim]
            # hidden = [2, batch, encoder_hidden_dim]
            return out, hidden, kl_loss
        else:
            out, hidden = self.mu(x)
            out, _ = pad_packed_sequence(out, batch_first=True)
            hidden = torch.cat((hidden[0], hidden[1]), dim=1)
            return out, hidden, Variable(torch.zeros(1).cuda(), requires_grad=False)

    def sample(self, mu, logvar):
        '''
        :param mu: [embedding_dim]
        :param logvar: [embedding_dim]
        :return: [embedding_dim]
        '''
        std = torch.exp(0.5 * logvar)
        eps = Variable(torch.randn(std.size()).cuda(), requires_grad=False)
        return mu + torch.mul(eps, std)

    def kl_convergence_loss(self, mu, logvar):
        return 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1 - logvar)


if __name__ == "__main__":
    input = [Variable(torch.LongTensor([3, 4, 2, 1, 0]).cuda()), Variable(torch.LongTensor([1, 2, 5]).cuda()),
             Variable(torch.LongTensor([5, 3]).cuda())]
    lengths = [5, 3, 2]
    model = Encoder(hidden=4, embedding_layer=nn.Embedding(6, 10)).cuda()
    out, hidden, kl_loss = model(input, lengths)
    print(out.size())
    print(hidden.size())
    print(kl_loss)
    print(lengths)
