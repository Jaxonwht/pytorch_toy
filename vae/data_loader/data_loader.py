import torch
from torch.utils.data import Dataset


class VAEData(Dataset):
    def __init__(self, filepath, vocab_file, max_seq_len, data_file_offset, vocab_file_offset):
        super().__init__()
        END_OF_STRING = "&EOS"
        START_OF_STRING = "&SOS"
        UNKNOWN = "&UNKNOWN"
        wordlists = [line.strip().split(" ")[vocab_file_offset:] for line in open(vocab_file)]
        self.index_dict = {0: START_OF_STRING, 1: END_OF_STRING, 2: UNKNOWN}
        self.word_dict = {START_OF_STRING: 0, END_OF_STRING: 1, UNKNOWN: 2}
        self.tag = []
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
                if line.startswith("d"):
                    tag = 0
                else:
                    tag = 1
                line = line.strip().split(" ")[data_file_offset:]
                line.append(END_OF_STRING)
                line.insert(0, START_OF_STRING)
                if len(line) > max_seq_len:
                    continue
                for index, token in enumerate(line):
                    if token not in self.word_dict:
                        line[index] = 2
                    else:
                        line[index] = self.word_dict[token]
                self.content.append(line)
                self.tag.append(tag)

    def __getitem__(self, item):
        return torch.tensor(self.content[item])

    def get_tag(self, index):
        return self.tag[index]

    def __len__(self):
        return len(self.content)

    def get_vocab_size(self):
        return len(self.index_dict)

    def get_index(self, token):
        return self.word_dict[token]

    def get_token(self, index):
        return self.index_dict[index]
