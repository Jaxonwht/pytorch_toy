from torch.utils.data import Dataset
import torch

from word2vec.data_loader import END_OF_STRING
from word2vec.data_loader import EmailDataset
from word2vec.data_loader import START_OF_STRING

COMPLETE_FILE = "../data/classtrain.txt"
whole_data = EmailDataset(COMPLETE_FILE, 1)


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
        return torch.tensor(self.content[item])

    def __len__(self):
        return len(self.content)

    def get_vocab_size(self):
        return self.whole_data.get_number_of_tokens()
