from torch.utils.data import Dataset
import torch

END_OF_STRING = "&EOS"
START_OF_STRING = "&SOS"
UNKNOWN = "&UNKNOWN"


class EmailDataset(Dataset):
    def __init__(self, file_path, context_size):
        super().__init__()
        wordlists = [line.strip().split(" ")[1:] for line in open(file_path)]
        self.index_dict = {0: START_OF_STRING, 1: END_OF_STRING}
        self.word_dict = {START_OF_STRING: 0, END_OF_STRING: 1}
        index = 2
        for wordseq in wordlists:
            for word in wordseq:
                if word not in self.word_dict:
                    self.word_dict[word] = index
                    self.index_dict[index] = word
                    index += 1
        self.index_pair = []
        for wordseq in wordlists:
            wordseq.insert(0, START_OF_STRING)
            wordseq.append(END_OF_STRING)
            for i in range(len(wordseq)):
                for j in range(max(0, i - context_size), i):
                    self.index_pair.append(torch.LongTensor([self.word_dict[wordseq[i]], self.word_dict[wordseq[j]]]))
                for j in range(i + 1, min(len(wordseq), i + context_size + 1)):
                    self.index_pair.append(torch.LongTensor([self.word_dict[wordseq[i]], self.word_dict[wordseq[j]]]))

    def __len__(self):
        return len(self.index_pair)

    def __getitem__(self, index):
        return self.index_pair[index]

    def get_token(self, i):
        return self.index_dict[i]

    def get_number_of_tokens(self):
        return len(self.word_dict)

    def get_index(self, token):
        if token not in self.word_dict:
            return -1
        return self.word_dict[token]
