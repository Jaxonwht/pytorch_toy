import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self, hidden, embedding_layer, num_layers=1, bidirectional=True):
        super().__init__()
        self.embedding = embedding_layer
        embed = embedding_layer.weight.size()[1]
        self.mu = nn.GRU(input_size=embed, hidden_size=hidden, num_layers=num_layers,
                         bidirectional=bidirectional, batch_first=True)
        self.logvar = nn.GRU(input_size=embed, hidden_size=hidden, num_layers=num_layers, bidirectional=bidirectional,
                             batch_first=True)

    def forward(self, x, lengths, variation):
        '''
        :param input: list of variable lengths of tensors, len(list) = batch_size, each entry in one element of the batch is [variable_seq_len]
        :param lengths: [batch_size]
        :return: (out, hidden, kl_loss), out is [batch, padded_seq_len, 2 x encoder_hidden_dim], hidden is [batch, 2 x encoder_hidden_dim], kl_loss is scalar
        '''
        if variation:
            input = Variable(torch.zeros(len(lengths), lengths[0], self.embedding.weight.size()[1]).cuda())
            # input = [batch, max_seq_len, embed]
            x = [self.embedding(token) for token in x]
            for seq in range(len(lengths)):
                input[seq, :lengths[seq], :] = x[seq]
            x = pack_padded_sequence(input, lengths=lengths, batch_first=True)
            mu_out, mu_hidden = self.mu(x)
            logvar_out, logvar_hidden = self.logvar(x)
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
            kl_loss = kl_loss.add(self.kl_convergence_loss(mu_hidden, logvar_hidden)).div(len(lengths))
            hidden = self.sample(mu_hidden, logvar_hidden)
            # out = [batch, max_seq_len, 2 x encoder_hidden_dim]
            # hidden = [2, batch, encoder_hidden_dim]
            hidden = torch.cat((hidden[0], hidden[1]), dim=1)
            # hidden = [batch, 2 x encoder_hidden_dim]
            return out, hidden, kl_loss
        else:
            input = Variable(torch.zeros(len(lengths), lengths[0], self.embedding.weight.size()[1]).cuda())
            x = [self.embedding(token) for token in x]
            for seq in range(len(lengths)):
                input[seq, :lengths[seq], :] = x[seq]
            x = pack_padded_sequence(input, lengths=lengths, batch_first=True)
            out, hidden = self.mu(x)
            return pad_packed_sequence(out, batch_first=True), hidden, torch.zeros(1)


    def sample(self, mu, logvar):
        '''
        :param mu: [embedding_dim]
        :param logvar: [embedding_dim]
        :return: [embedding_dim]
        '''
        std = torch.exp(0.5 * logvar)
        eps = Variable(torch.randn(std.size()).cuda())
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
