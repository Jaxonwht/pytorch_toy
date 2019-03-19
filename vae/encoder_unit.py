import torch.nn as nn
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
    def __init__(self, embed, hidden, num_layers, bidirectional):
        super().__init__()
        self.gru = nn.GRU(input_size=embed, hidden_size=hidden, batch_first=True, num_layers=num_layers,
                          bidirectional=bidirectional)

    def forward(self, x):
        out, hidden = self.gru(x)
        return hidden


class VAEData(Dataset):
    def __init__(self, filepath):
        super().__init__()
        self.whole_data = EmailDataset(COMPLETE_FILE, 1)
        self.content = []
        with open(filepath) as f:
            for line in f:
                line = line.strip().split(" ")[1:]
                line.append(END_OF_STRING)
                line.insert(0, START_OF_STRING)
                add = True
                for index, token in enumerate(line):
                    num = self.whole_data.get_index(token)
                    if num == -1:
                        add = False
                        break
                    line[index] = num
                if add:
                    self.content.append(line)

    def __getitem__(self, item):
        return self.content[item]

    def __len__(self):
        return len(self.content)


if __name__ == "__main__":
    data = VAEData("../data/democratic_only.train.en")
    print(len(data))
    print(data[0])
    print(data[1])