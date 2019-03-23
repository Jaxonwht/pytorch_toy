import torch.nn as nn
import torch
from vae.main_module.attention_unit import Attention
from torch.nn.utils.rnn import pad_sequence

class Decoder(nn.Module):
    def __init__(self, embed, encoder_hidden, decoder_hidden, vocab_size):
        super().__init__()
        self.embed = embed
        self.gru_cell = nn.GRUCell(input_size=embed + 2 * encoder_hidden, hidden_size=decoder_hidden)
        self.attention = Attention(encoder_hidden_dim=encoder_hidden, decoder_hidden_dim=decoder_hidden)
        self.interpret = nn.Linear(in_features=decoder_hidden, out_features=vocab_size)

    def forward(self, input, hidden, encoder_outs, lengths, teacher_forcing_ratio=0.5):
        '''
        :param input: list, len(input) = batch. Each entry is [variable_seq_len, embed_dim]
        :param hidden: [decoder_hidden_dim, batch]
        :param encoder_outs: [batch, max_seq_len, 2 x encoder_hidden_dim]
        :param lengths: [batch]
        :param teacher_forcing_ratio: scalar
        '''
        paddded_input = pad_sequence(input, batch_first=True)
        # padded_input = [batch, max_seq_len, embed]
        outs = torch.zeros(len(lengths), encoder_outs.size()[1], self.embed)
        # outs = [batch, max_seq_len, embed_dim]
        outs[:, 0, :] = paddded_input[:, 0, :]
        for index in range(1, encoder_outs.size()[1]):
            context = self.attention(encoder_outs, hidden, lengths)
            # context = [batch, 2 x encoder_hidden_dim]
            if torch.rand(1) < teacher_forcing_ratio:
                selected = input[:, index - 1, :]
            else:
                selected = outs[:, index - 1, :]
            # selected = [batch, embed]
            modified_input = torch.cat((selected, context), dim=1)
            # modified_input = [batch, embed + 2 x encoder_hidden_dim]
            hidden = self.gru_cell(modified_input, hidden.t())
            outs[:, index, :] = hidden
        return outs

if __name__ == "__main__":
    model = Decoder(3, 2, 2, 10)
    input = [torch.randn(5, 3), torch.randn(4, 3), torch.randn(2, 3)]
    lengths = torch.tensor([5, 4, 2])
    hidden = torch.randn(2, 3)
    encoder_outs = torch.randn(3, 5, 2 * 2)
    outs = model(input, hidden, encoder_outs, lengths)
    print(outs.size())



