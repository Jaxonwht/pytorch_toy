import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

class EmailDataset(Dataset):
    def __init__(self, file_path, context_size):
        super().__init__()
        wordlists = [line.strip().split(" ")[1:] for line in open(file_path)]
        self.index_dict = {}
        self.word_dict = {}
        for wordseq in wordlists:
            for word in wordseq:
                if word not in self.index_dict:
                    self.word_dict[word] = len(self.index_dict)
                    self.index_dict[len(self.index_dict)] = word
        self.index_pair = []
        for wordseq in wordlists:
            for i in range(len(wordseq)):
                for j in range(max(0, i - context_size), i):
                    self.index_pair.append((self.word_dict[wordseq[i]], self.word_dict[wordseq[j]]))
                for j in range(i + 1, min(len(wordseq), i + context_size + 1)):
                    self.index_pair.append((self.word_dict[wordseq[i]], self.word_dict[wordseq[j]]))

    def __len__(self):
        return len(self.index_pair)

    def __getitem__(self, index):
        return self.index_pair[index]

    def getToken(self, i):
        return self.index_dict[i]


