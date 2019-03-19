import torch.nn as nn
from word2vec.word_to_vec import Word2Vec
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import Dataset
from word2vec.data_loader import START_OF_STRING
from word2vec.data_loader import END_OF_STRING
from word2vec.data_loader import EmailDataset


COMPLETE_FILE = "../data/classtrain.txt"
whole_data = EmailDataset(COMPLETE_FILE, 1)

def convert_batch_to_sorted_list(x):
    list_of_sequences = [x[i] for i in range(len(x))]
    list_of_sequences.sort(key=lambda seq: len(seq), reverse=True)
    return pack_sequence(list_of_sequences)


class Encoder(nn.Module):
    def __init__(self, vocabulary, embed, hidden, num_layers, bidirectional):
        super().__init__()
        self.gru = nn.GRU(input_size=embed, hidden_size=hidden, batch_first=True, num_layers=num_layers,
                          bidirectional=bidirectional)

    def forward(self, x):
        out, hidden = self.gru(x)
        return hidden


class VAEData(filepath)