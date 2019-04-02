import sys

sys.path.append('/dscrhome/hw186/pytorch_toy')
import torch
from torch.utils.data import Dataset

class VAEData(Dataset):
    def __init__(self, filepath, vocab_file, max_seq_len, data_file_offset, vocab_file_offset):
        super().__init__()
        END_OF_STRING = "&EOS"
        START_OF_STRING = "&SOS"
        wordlists = [line.strip().split(" ")[vocab_file_offset:] for line in open(vocab_file)]
        self.index_dict = {0: START_OF_STRING, 1: END_OF_STRING}
        self.word_dict = {START_OF_STRING: 0, END_OF_STRING: 1}
        index = 2
        for wordseq in wordlists:
            for word in wordseq:
                if word not in self.word_dict:
                    self.word_dict[word] = index
                    self.index_dict[index] = word
                    index += 1
        self.content = []
        with open(filepath) as f:
            for line in f:
                line = line.strip().split(" ")[data_file_offset:]
                line.append(END_OF_STRING)
                line.insert(0, START_OF_STRING)
                if len(line) > max_seq_len:
                    continue
                add = True
                for index, token in enumerate(line):
                    if token not in self.word_dict:
                        add = False
                        break
                    num = self.word_dict[token]
                    line[index] = num
                if add:
                    self.content.append(line)

    def __getitem__(self, item):
        return torch.LongTensor(self.content[item])

    def __len__(self):
        return len(self.content)

    def get_vocab_size(self):
        return len(self.word_dict)

    def get_index(self, token):
        return self.word_dict[token]

    def get_token(self, index):
        return self.index_dict[index]
