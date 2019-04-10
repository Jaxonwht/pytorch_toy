import sys

sys.path.append("/workspace/pytorch_toy/")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from vae.data_loader.data_loader import VAEData


class Classifier(nn.Module):
    def __init__(self, vocab_size, rnn_hidden_dim, mid_hidden_dim1, mid_hidden_dim2, class_number):
        super().__init__()
        self.activation = nn.LeakyReLU()
        self.rnn = nn.GRU(input_size=vocab_size, hidden_size=rnn_hidden_dim, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(in_features=2 * rnn_hidden_dim, out_features=mid_hidden_dim1)
        self.fc2 = nn.Linear(in_features=mid_hidden_dim1, out_features=mid_hidden_dim2)
        self.fc3 = nn.Linear(in_features=mid_hidden_dim2, out_features=class_number)

    def forward(self, input):
        _, hidden = self.rnn(input)
        # hidden = [2 x num_layers, batch, rnn_hidden_dim]
        hidden = hidden.permute(1, 0, 2)
        # hidden = [batch, 2 x num_layers, rnn_hidden_dim]
        return self.fc3(self.activation(self.fc2(self.activation(self.fc1(hidden.flatten(start_dim=1))))))

    def untrain(self):
        for param in self.parameters():
            param.requires_grad = False


if __name__ == "__main__":
    if torch.cuda.is_available():
        my_device = torch.device("cuda")
    else:
        my_device = torch.device("cpu")
    BATCH_SIZE = 50
    MAX_SEQ_LEN = 50
    LEARNING_RATE = 1e-2
    EPOCHS = 300
    HIDDEN_DIM = 50
    MID_HIDDEN_1 = 50
    MID_HIDDEN_2 = 10
    VOCAB = "../data/vocab.txt"
    TRAINING = "../data/mixed_train.txt"
    TESTING = "../data/democratic_only.dev.en"
    PRETRAINED_MODEL_FILE_PATH = "model/checkpoint.pt"
    MODEL_FILE_PATH = "model/checkpoint.pt"
    pretrained = False
    training = True
    variation = False

    if training:
        training_data = VAEData(filepath=TRAINING, vocab_file=VOCAB, max_seq_len=MAX_SEQ_LEN, data_file_offset=1,
                                vocab_file_offset=1)
        model = Classifier(vocab_size=training_data.get_vocab_size(), rnn_hidden_dim=HIDDEN_DIM,
                           mid_hidden_dim1=MID_HIDDEN_1, mid_hidden_dim2=MID_HIDDEN_2, class_number=2).to(my_device)
        optim = Adam(model.parameters(), lr=LEARNING_RATE)
        if pretrained:
            model.load_state_dict(torch.load(PRETRAINED_MODEL_FILE_PATH)["model_state_dict"])
            optim.load_state_dict(torch.load(PRETRAINED_MODEL_FILE_PATH)["optimizer_state_dict"])
        loss_fn = nn.CrossEntropyLoss()
        for e in range(EPOCHS):
            for batch in range(len(training_data) // BATCH_SIZE):
                raw = [
                    (training_data[batch * BATCH_SIZE + i].to(my_device), training_data.get_tag(batch * BATCH_SIZE + i))
                    for i in range(BATCH_SIZE)]
                raw.sort(key=lambda x: len(x[0]), reverse=True)
                input = []
                for i in range(BATCH_SIZE):
                    temp = raw[i][0]
                    # temp = [seq_len]
                    one_hot_encoding = torch.zeros(len(temp), training_data.get_vocab_size(), device=my_device)
                    # one_hot_encoding = [seq_len, vocab_size]
                    one_hot_encoding.scatter_(1, temp.unsqueeze(1), 1)
                    input.append(one_hot_encoding)
                target = torch.tensor([x[1] for x in raw], device=my_device, requires_grad=False)
                pack = nn.utils.rnn.pack_sequence(input).to(my_device)
                scores = model(pack)
                loss = loss_fn(scores, target)
                optim.zero_grad()
                loss.backward()
                optim.step()
                if batch % 10 == 0:
                    print("Epoch {}, Batch {}, Loss {}".format(e, batch, loss.item()))
            torch.save(
                {"Epoch": e,
                 "Loss": loss.item(),
                 "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optim.state_dict()}, MODEL_FILE_PATH)
    else:
        training_data = VAEData(filepath=TRAINING, vocab_file=VOCAB, max_seq_len=MAX_SEQ_LEN, data_file_offset=1,
                                vocab_file_offset=1)
        model = Classifier(vocab_size=training_data.get_vocab_size(), rnn_hidden_dim=HIDDEN_DIM,
                           mid_hidden_dim1=MID_HIDDEN_1, mid_hidden_dim2=MID_HIDDEN_2, class_number=2).to(my_device)
        model.load_state_dict(torch.load(PRETRAINED_MODEL_FILE_PATH)["model_state_dict"])
        model.eval()
        for e in range(EPOCHS):
            for batch in range(len(training_data) // BATCH_SIZE):
                raw = [
                    (training_data[batch * BATCH_SIZE + i].to(my_device), training_data.get_tag(batch * BATCH_SIZE + i))
                    for i in range(BATCH_SIZE)]
                raw.sort(key=lambda x: len(x[0]), reverse=True)
                lengths = torch.tensor([len(x[0]) for x in raw], device=my_device)
                input = []
                for i in range(BATCH_SIZE):
                    temp = raw[i][0]
                    # temp = [seq_len]
                    one_hot_encoding = torch.zeros(len(temp), training_data.get_vocab_size(), device=my_device)
                    # one_hot_encoding = [seq_len, vocab_size]
                    one_hot_encoding.scatter_(1, temp.unsqueeze(1), 1)
                    input.append(one_hot_encoding)
                target = torch.tensor([x[1] for x in raw], device=my_device, requires_grad=False)
                pack = nn.utils.rnn.pack_sequence(input).to(my_device)
                scores = model(pack)
                print(F.softmax(scores))
