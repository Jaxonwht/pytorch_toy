from torch.utils.data import Dataset
import torch


class EmailDataset(Dataset):
    def __init__(self, file_path, context_size):
        super().__init__()
        wordlists = [line.strip().split(" ")[1:] for line in open(file_path)]
        self.index_dict = {}
        self.word_dict = {}
        index = 0
        for wordseq in wordlists:
            for word in wordseq:
                if word not in self.word_dict:
                    self.word_dict[word] = index
                    self.index_dict[index] = word
                    index += 1
        self.index_pair = []
        for wordseq in wordlists:
            for i in range(len(wordseq)):
                for j in range(max(0, i - context_size), i):
                    self.index_pair.append(torch.tensor([self.word_dict[wordseq[i]], self.word_dict[wordseq[j]]]))
                for j in range(i + 1, min(len(wordseq), i + context_size + 1)):
                    self.index_pair.append(torch.tensor([self.word_dict[wordseq[i]], self.word_dict[wordseq[j]]]))

    def __len__(self):
        return len(self.index_pair)

    def __getitem__(self, index):
        return self.index_pair[index]

    def get_token(self, i):
        return self.index_dict[i]

    def get_number_of_tokens(self):
        return len(self.word_dict)
