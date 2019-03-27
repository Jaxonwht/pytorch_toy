import sys

sys.path.append('/dscrhome/hw186/pytorch_toy')
import torch
from torch.utils.data import Dataset

from word2vec.data_loader.data_loader import END_OF_STRING
from word2vec.data_loader.data_loader import EmailDataset
from word2vec.data_loader.data_loader import START_OF_STRING


class VAEData(Dataset):
    def __init__(self, filepath, vocab_data_file, max_seq_len):
        super().__init__()
        self.whole_data = EmailDataset(vocab_data_file, 0)
        self.content = []
        with open(filepath) as f:
            for line in f:
                line = line.strip().split(" ")[1:]
                line.append(END_OF_STRING)
                line.insert(0, START_OF_STRING)
                if len(line) > max_seq_len:
                    continue
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
        return torch.LongTensor(self.content[item])

    def __len__(self):
        return len(self.content)

    def get_vocab_size(self):
        return self.whole_data.get_number_of_tokens()