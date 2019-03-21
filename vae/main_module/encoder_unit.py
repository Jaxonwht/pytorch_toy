import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence

from vae.data_loader.data_loader import VAEData


class Encoder(nn.Module):
    def __init__(self, vocab, embed, hidden, num_layers=1, bidirectional=True, embedding_weights=None, max_seq_len = 50):
        super().__init__()
        self.max_seq_len = max_seq_len
        if embedding_weights:
            self.word_embed = nn.Embedding.from_pretrained(embeddings=embedding_weights)
        else:
            self.word_embed = nn.Embedding(vocab, embed)
        self.gru = nn.GRU(input_size=embed, hidden_size=hidden, num_layers=num_layers,
                          bidirectional=bidirectional, batch_first=True)

    def forward(self, input, lengths):
        '''
        :param x: list of tensors of variable lengths, len(list) = batch_size
        :param lengths: [batch_size]
        :return: (out, hidden), out is [batch, padded_seq_len, 2 x encoder_hidden_dim], hidden is [batch, 2 x encoder_hidden_dim]
        '''
        x = [self.word_embed(seq) for seq in input]
        x = pad_packed_sequence(x, batch_first=True, total_length=self.max_seq_len)
        # x = [batch_size, padded_seq_len, embedding_dim]
        out, hidden = self.gru(x)
        # out is a padded seq
        # hidden = [2 x encoder_hidden_dim, batch]
        # hidden can be separated into part = hidden.view(num_layers, num_directions, batch, hidden_size)
        return out, hidden


if __name__ == "__main__":
    data = VAEData("../../data/democratic_only.train.en")
    print(len(data))
    print(data[0])
    print(data[1])
    rnn = nn.GRU(10, 20, 2)
    input = torch.randn(5, 3, 10)
    h0 = torch.randn(2, 3, 20)
    output, hn = rnn(input, h0)
