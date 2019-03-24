import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from vae.main_module.attention_unit import Attention


class Decoder(nn.Module):
    def __init__(self, encoder_hidden, decoder_hidden, embedding_layer):
        super().__init__()
        self.embedding = embedding_layer
        vocab_size = embedding_layer.weight.size()[0]
        embed = embedding_layer.weight.size()[1]
        self.gru_cell = nn.GRUCell(input_size=embed + 2 * encoder_hidden, hidden_size=decoder_hidden)
        self.attention = Attention(encoder_hidden_dim=encoder_hidden, decoder_hidden_dim=decoder_hidden)
        self.interpret = nn.Linear(in_features=decoder_hidden, out_features=vocab_size)
        self.token_activation = nn.LeakyReLU()

    def forward(self, input, hidden, encoder_outs, lengths, teacher_forcing_ratio=0.5):
        '''
        :param input: list, len(input) = batch. Each entry is [variable_seq_len]
        :param hidden: [batch, hidden]
        :param encoder_outs: [batch, max_seq_len, 2 x encoder_hidden_dim]
        :param lengths: [batch]
        :param teacher_forcing_ratio: scalar
        '''
        reconstruction_loss = torch.zeros(1)
        input = [self.embedding(token) for token in input]
        paddded_input = pad_sequence(input, batch_first=True)
        # padded_input = [batch, max_seq_len, embed]
        out = torch.zeros(len(lengths), encoder_outs.size()[1], self.embedding.weight.size()[0])
        # out = [batch, max_seq_len, vocab_size]
        out[:, 0, 0] = torch.ones(len(lengths))
        for index in range(1, encoder_outs.size()[1]):
            context = self.attention(encoder_outs, hidden, lengths)
            # context = [batch, 2 x encoder_hidden_dim]
            if torch.rand(1) < teacher_forcing_ratio:
                selected = paddded_input[:, index - 1, :]
            else:
                selected = self.embedding(torch.argmax(out[:, index - 1, :], dim=1))
            # selected = [batch, embed]
            modified_input = torch.cat((selected, context), dim=1)
            # modified_input = [batch, embed + 2 x encoder_hidden_dim]
            hidden = self.gru_cell(modified_input, hidden)
            out[:, index, :] = self.token_activation(self.interpret(hidden))
        return out


if __name__ == "__main__":
    model = Decoder(3, 2, nn.Embedding(7, 10))
    input = [torch.randint(7, (5,)), torch.randint(7, (4,)), torch.randint(7, (2,))]
    lengths = torch.tensor([5, 4, 2])
    hidden = torch.randn(3, 2)
    encoder_outs = torch.randn(3, 5, 2 * 3)
    outs = model(input, hidden, encoder_outs, lengths)
    print(outs.size())
