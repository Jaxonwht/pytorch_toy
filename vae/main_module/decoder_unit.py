import torch.nn as nn
from vae.main_module.attention_unit import Attention

class Decoder(nn.Module):
    def __init__(self, embed, encoder_hidden, decoder_hidden, vocab_size):
        super().__init__()
        self.gru_cell = nn.GRUCell(input_size=embed, hidden_size=decoder_hidden)
        self.attention = Attention(encoder_hidden_dim=encoder_hidden, decoder_hidden_dim=decoder_hidden)
        self.interpret = nn.Linear(in_features=decoder_hidden, out_features=vocab_size)

    def forward(self, input, hidden, encoder_outs, lengths, teacher_forcing_ratio=0):
        '''
        :param input: list, len(input) = batch. Each entry is [variable_seq_len, embed_dim]
        :param hidden: [decoder_hidden_dim, batch]
        :param encoder_outs: [batch, max_seq_len, 2 x encoder_hidden_dim]
        :param lengths: [batch]
        :param teacher_forcing_ratio: scalar
        '''
        for seq_len in range(encoder_outs.size()[1]):
            context = self.attention(encoder_outs, hidden, lengths)
            # context = [batch, 2 x encoder_hidden_dim]

