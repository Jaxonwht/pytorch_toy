import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from vae.main_module.attention_unit import Attention


class Decoder(nn.Module):
    def __init__(self, encoder_hidden, decoder_hidden, embedding_layer, device):
        super().__init__()
        self.device = device
        self.embedding = embedding_layer
        vocab_size = embedding_layer.weight.size()[0]
        embed = embedding_layer.weight.size()[1]
        self.gru_cell = nn.GRUCell(input_size=embed + 2 * encoder_hidden, hidden_size=decoder_hidden)
        self.attention = Attention(encoder_hidden_dim=encoder_hidden, decoder_hidden_dim=decoder_hidden, device=device)
        self.interpret = nn.Linear(in_features=decoder_hidden, out_features=vocab_size)

    def forward(self, input, hidden, encoder_outs, lengths, teacher_forcing_ratio):
        '''
        :param input: list, len(input) = batch. Each entry is [variable_seq_len]
        :param hidden: [batch, hidden]
        :param encoder_outs: [batch, max_seq_len, 2 x encoder_hidden_dim]
        :param lengths: [batch]
        :param teacher_forcing_ratio: scalar
        '''
        input = [self.embedding(token) for token in input]
        padded_input = pad_sequence(input, batch_first=True)
        # padded_input = [batch, max_seq_len, embed]
        out = torch.zeros(len(lengths), encoder_outs.size()[1], self.embedding.weight.size()[0], device=self.device)
        # out = [batch, max_seq_len, vocab_size]
        out[:, 0, 0] = torch.ones(len(lengths))
        for index in range(1, encoder_outs.size()[1]):
            context = self.attention(encoder_outs, hidden, lengths)
            # context = [batch, 2 x encoder_hidden_dim]
            if torch.rand(1) < teacher_forcing_ratio:
                selected = padded_input[:, index - 1, :]
            else:
                selected = self.embedding(torch.argmax(out[:, index - 1, :], dim=1).to(self.device))
            # selected = [batch, embed]
            modified_input = torch.cat((selected, context), dim=1)
            # modified_input = [batch, embed + 2 x encoder_hidden_dim]
            hidden = self.gru_cell(modified_input, hidden)
            out[:, index, :] = self.interpret(hidden)
        return out

    def inference(self, initial_hidden, encoder_outs, beam_width):
        '''
        :param initial_hidden: [1, decoder_hidden_dim]
        :param encoder_outs: [1, seq_len, 2 x encoder_hidden_dim]
        :param beam_width: scalar
        :return: [some_other_seq_len]
        '''


if __name__ == "__main__":
    if torch.cuda.is_available():
        my_device = torch.device("cpu")
    else:
        my_device = torch.device("gpu")
    model = Decoder(3, 2, nn.Embedding(7, 10), my_device)
    input = [torch.randint(7, (5,)), torch.randint(7, (4,)), torch.randint(7, (2,))]
    lengths = torch.tensor([5, 4, 2])
    hidden = torch.randn(3, 2)
    encoder_outs = torch.randn(3, 5, 2 * 3)
    outs = model(input, hidden, encoder_outs, lengths)
    print(outs[:, 0, :])
