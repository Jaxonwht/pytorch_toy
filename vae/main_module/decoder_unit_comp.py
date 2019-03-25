import sys

sys.path.append('/dscrhome/hw186/pytorch_toy')
import torch
import torch.nn as nn
from torch.autograd import Variable

from vae.main_module.attention_unit_comp import Attention


class Decoder(nn.Module):
    def __init__(self, encoder_hidden, decoder_hidden, embedding_layer, leaky_relu_constant=0.1):
        super().__init__()
        self.embedding = embedding_layer
        vocab_size = embedding_layer.weight.size()[0]
        embed = embedding_layer.weight.size()[1]
        self.gru_cell = nn.GRUCell(input_size=embed + 2 * encoder_hidden, hidden_size=decoder_hidden)
        self.attention = Attention(encoder_hidden_dim=encoder_hidden, decoder_hidden_dim=decoder_hidden)
        self.interpret = nn.Linear(in_features=decoder_hidden, out_features=vocab_size)
        self.token_activation = nn.LeakyReLU(leaky_relu_constant)

    def forward(self, input, hidden, encoder_outs, lengths, teacher_forcing_ratio=0.5):
        '''
        :param input: list, len(input) = batch. Each entry is [variable_seq_len]
        :param hidden: [batch, hidden]
        :param encoder_outs: [batch, max_seq_len, 2 x encoder_hidden_dim]
        :param lengths: [batch]
        :param teacher_forcing_ratio: scalar
        '''
        input = [self.embedding(token) for token in input]
        padded_input = Variable(torch.zeros(len(lengths), lengths[0], self.embedding.weight.size()[1]).cuda())
        # input = [batch, max_seq_len, embed]
        for seq in range(len(lengths)):
            padded_input[seq, :lengths[seq], :] = input[seq]
        # padded_input = [batch, max_seq_len, embed]
        out = Variable(torch.zeros(len(lengths), encoder_outs.size()[1], self.embedding.weight.size()[0]).cuda())
        # out = [batch, max_seq_len, vocab_size]
        out[:, 0, 0] = torch.ones(len(lengths))
        for index in range(1, encoder_outs.size()[1]):
            context = self.attention(encoder_outs, hidden, lengths)
            # context = [batch, 2 x encoder_hidden_dim]
            if torch.rand(1)[0] < teacher_forcing_ratio:
                selected = padded_input[:, index - 1, :]
            else:
                selected = self.embedding(torch.max(out[:, index - 1, :], dim=1)[1])
            # selected = [batch, embed]
            modified_input = torch.cat((selected, context), dim=1)
            # modified_input = [batch, embed + 2 x encoder_hidden_dim]
            hidden = self.gru_cell(modified_input, hidden)
            out[:, index, :] = self.token_activation(self.interpret(hidden))
        return out


if __name__ == "__main__":
    model = Decoder(3, 2, nn.Embedding(7, 10)).cuda()
    input = [Variable(torch.LongTensor([4, 2, 0, 1, 6]).cuda()), Variable(torch.LongTensor([6, 3, 1, 2]).cuda()),
             Variable(torch.LongTensor([4, 5]).cuda())]
    lengths = [5, 4, 2]
    hidden = Variable(torch.randn(3, 2).cuda())
    encoder_outs = Variable(torch.randn(3, 5, 2 * 3).cuda())
    outs = model(input, hidden, encoder_outs, lengths)
    print(outs[:, 1, :])
