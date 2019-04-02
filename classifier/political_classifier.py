import sys

sys.path.append('/dscrhome/hw186/pytorch_toy')
import torch
import torch.nn as nn
from torch.optim import Adam

from vae.data_loader.data_loader import VAEData


class Classifier(nn.Module):
    def __init__(self, vocab_size, rnn_hidden_dim, rnn_layers, mid_hidden_dim, class_number):
        super().__init__()
        self.encoder = nn.GRU(input_size=vocab_size, hidden_size=rnn_hidden_dim, bidirectional=True,
                              num_layers=rnn_layers, batch_first=True)
        self.conv1 = nn.Conv1d(in_channels=rnn_layers * 2, out_channels=10, kernel_size=rnn_hidden_dim // 5)
        self.conv2 = nn.Conv1d(in_channels=10, out_channels=50, kernel_size=rnn_hidden_dim // 10)
        self.conv3 = nn.Conv1d(in_channels=50, out_channels=200, kernel_size=rnn_hidden_dim // 100)
        self.activation = nn.LeakyReLU()
        self.fc1 = nn.Linear(
            in_features=(rnn_hidden_dim - rnn_hidden_dim // 5 - rnn_hidden_dim // 10 - rnn_hidden_dim // 100 + 3) * 200,
            out_features=mid_hidden_dim)
        self.fc2 = nn.Linear(in_features=mid_hidden_dim, out_features=class_number)

    def forward(self, *input):
        _, hidden = self.encoder(*input)
        # hidden = [2 x num_layers, batch, rnn_hidden_dim]
        hidden = hidden.permute(1, 0, 2)
        # hidden = [batch, 2 x num_layers, rnn_hidden_dim]
        features = self.activation(self.conv3(self.activation(self.conv2(self.activation(self.conv1(hidden))))))
        return self.fc2(self.activation(self.fc1(features.flatten(1))))


if __name__ == "__main__":
    if torch.cuda.is_available():
        my_device = torch.device("cuda")
    else:
        my_device = torch.device("cpu")
    BATCH_SIZE = 50
    MAX_SEQ_LEN = 50
    RNN_HIDDEN_DIM = 300
    LEARNING_RATE = 1e-2
    EPOCHS = 300
    MID_HIDDEN = 50
    RNN_LAYERS = 2
    VOCAB = "../data/classtrain.txt"
    TRAINING = "../data/mixed_train.txt"
    TESTING = "../data/democratic_only.test.en"
    PRETRAINED_MODEL_FILE_PATH = "model/checkpoint.pt"
    MODEL_FILE_PATH = "model/checkpoint.pt"
    pretrained = False

    training_data = VAEData(filepath=TRAINING, vocab_file=VOCAB, max_seq_len=MAX_SEQ_LEN, data_file_offset=1,
                            vocab_file_offset=1)
    model = Classifier(vocab_size=training_data.get_vocab_size(), rnn_hidden_dim=RNN_HIDDEN_DIM, rnn_layers=2,
                       mid_hidden_dim=MID_HIDDEN, class_number=2).to(my_device)
    optim = Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    for e in range(EPOCHS):
        for batch in range(len(training_data) // BATCH_SIZE):
            input = []
            target = []
            for i in range(BATCH_SIZE):
                temp = training_data[batch * BATCH_SIZE + i].to(my_device)
                # temp = [seq_len]
                dummy = torch.zeros(len(temp), training_data.get_vocab_size(), device=my_device)
                # dummy = [seq_len, vocab_size]
                dummy.scatter_(1, temp.unsqueeze(1), 1)
                input.append(dummy)
                target.append((training_data.get_tag(batch * BATCH_SIZE + i), len(dummy)))
            target.sort(key=lambda x: x[1], reverse=True)
            target = torch.tensor([x[0] for x in target], device=my_device)
            input.sort(key=lambda seq: len(seq), reverse=True)
            pack = nn.utils.rnn.pack_sequence(input).to(my_device)
            scores = model(pack)
            loss = loss_fn(scores, target)
            optim.zero_grad()
            loss.backward()
            optim.step()
            if batch % 10 == 0:
                print("Epoch {}, Batch {}, Loss {}".format(e, batch, loss.item()))
