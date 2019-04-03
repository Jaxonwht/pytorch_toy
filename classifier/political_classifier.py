import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from vae.data_loader.data_loader import VAEData


class Classifier(nn.Module):
    def __init__(self, vocab_size, rnn_hidden_dim, rnn_layers, mid_hidden_dim1, mid_hidden_dim2, class_number):
        super().__init__()
        self.encoder = nn.GRU(input_size=vocab_size, hidden_size=rnn_hidden_dim, bidirectional=True,
                              num_layers=rnn_layers, batch_first=True)
        self.activation = nn.LeakyReLU()
        self.fc1 = nn.Linear(in_features=rnn_layers * 2 * rnn_hidden_dim, out_features=mid_hidden_dim1)
        self.fc2 = nn.Linear(in_features=mid_hidden_dim1, out_features=mid_hidden_dim2)
        self.fc3 = nn.Linear(in_features=mid_hidden_dim2, out_features=class_number)

    def forward(self, *input):
        _, hidden = self.encoder(*input)
        # hidden = [2 x num_layers, batch, rnn_hidden_dim]
        hidden = hidden.permute(1, 0, 2)
        # hidden = [batch, 2 x num_layers, rnn_hidden_dim]
        return self.fc3(self.activation(self.fc2(self.activation(self.fc1(hidden.flatten(1))))))

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
    RNN_HIDDEN_DIM = 100
    LEARNING_RATE = 1e-2
    EPOCHS = 300
    MID_HIDDEN_1 = 100
    MID_HIDDEN_2 = 40
    RNN_LAYERS = 1
    VOCAB = "../data/classtrain.txt"
    TRAINING = "../data/mixed_train.txt"
    TESTING = "../data/democratic_only.dev.en"
    PRETRAINED_MODEL_FILE_PATH = "model/checkpoint.pt"
    MODEL_FILE_PATH = "model/checkpoint.pt"
    pretrained = True
    training = True

    if training:
        training_data = VAEData(filepath=TRAINING, vocab_file=VOCAB, max_seq_len=MAX_SEQ_LEN, data_file_offset=1,
                                vocab_file_offset=1)
        model = Classifier(vocab_size=training_data.get_vocab_size(), rnn_hidden_dim=RNN_HIDDEN_DIM,
                           rnn_layers=RNN_LAYERS,
                           mid_hidden_dim1=MID_HIDDEN_1, mid_hidden_dim2=MID_HIDDEN_2, class_number=2).to(my_device)
        optim = Adam(model.parameters(), lr=LEARNING_RATE)
        if pretrained:
            model.load_state_dict(torch.load(PRETRAINED_MODEL_FILE_PATH)["model_state_dict"])
            optim.load_state_dict(torch.load(PRETRAINED_MODEL_FILE_PATH)["optimizer_state_dict"])
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
            torch.save(
                {"Epoch": e,
                 "Loss": loss.item(),
                 "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optim.state_dict()}, MODEL_FILE_PATH)
    else:
        training_data = VAEData(filepath=TESTING, vocab_file=VOCAB, max_seq_len=MAX_SEQ_LEN, data_file_offset=1,
                                vocab_file_offset=1)
        model = Classifier(vocab_size=training_data.get_vocab_size(), rnn_hidden_dim=RNN_HIDDEN_DIM,
                           rnn_layers=RNN_LAYERS,
                           mid_hidden_dim1=MID_HIDDEN_1, mid_hidden_dim2=MID_HIDDEN_2, class_number=2).to(my_device)
        model.eval()
        if pretrained:
            model.load_state_dict(
                torch.load(PRETRAINED_MODEL_FILE_PATH, map_location=lambda storage, loc: storage)["model_state_dict"])
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
                print(F.softmax(scores, dim=1))
