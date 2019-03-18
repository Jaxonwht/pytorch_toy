import torch.nn as nn
from word2vec.word_to_vec import Word2Vec


class Encoder(nn.Module):
    def __init__(self, seq_len, vocabulary, embed, hidden, num_layers, bidirectional):
        super().__init__()
        self.embed = Word2Vec(vocabulary, embed)
        self.gru = nn.GRU(input_size=embed, hidden_size=hidden, batch_first=True, num_layers=num_layers, bidirectional=bidirectional)


