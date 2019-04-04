import sys

sys.path.append('/dscrhome/hw186/pytorch_toy')
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam

from vae.data_loader.data_loader_comp import VAEData
from vae.main_module.variational_autoencoder_comp import VAE


class Classifier(nn.Module):
    def __init__(self, rnn_hidden_dim, mid_hidden_dim1, mid_hidden_dim2, class_number):
        super().__init__()
        self.activation = nn.LeakyReLU()
        self.fc1 = nn.Linear(in_features=2 * rnn_hidden_dim, out_features=mid_hidden_dim1)
        self.fc2 = nn.Linear(in_features=mid_hidden_dim1, out_features=mid_hidden_dim2)
        self.fc3 = nn.Linear(in_features=mid_hidden_dim2, out_features=class_number)

    def forward(self, hidden):
        # hidden = [batch, 2 x rnn_hidden_dim]
        return self.fc3(self.activation(self.fc2(self.activation(self.fc1(hidden)))))

    def untrain(self):
        for param in self.parameters():
            param.requires_grad = False


if __name__ == "__main__":
    BATCH_SIZE = 50
    MAX_SEQ_LEN = 50
    ENCODER_HIDDEN_SIZE = 300
    DECODER_HIDDEN_SIZE = 300
    LEARNING_RATE = 1e-2
    EPOCHS = 300
    EMBEDDING_SIZE = 500
    MID_HIDDEN_1 = 100
    MID_HIDDEN_2 = 40
    VOCAB = "../data/classtrain.txt"
    TRAINING = "../data/mixed_train.txt"
    TESTING = "../data/democratic_only.dev.en"
    VAE_MODEL_FILE_PATH = "../vae/model/checkpoint.pt"
    PRETRAINED_MODEL_FILE_PATH = "model/checkpoint.pt"
    MODEL_FILE_PATH = "model/checkpoint.pt"
    pretrained = False
    variation = False

    training_data = VAEData(filepath=TRAINING, vocab_file=VOCAB, max_seq_len=MAX_SEQ_LEN, data_file_offset=1,
                            vocab_file_offset=1)
    vae_model = VAE(embed=EMBEDDING_SIZE, encoder_hidden=ENCODER_HIDDEN_SIZE,
                    decoder_hidden=DECODER_HIDDEN_SIZE,
                    vocabulary=training_data.get_vocab_size()).cuda()
    vae_model.load_state_dict(torch.load(VAE_MODEL_FILE_PATH)["model_state_dict"], strict=False)
    embedding = vae_model.embedding
    encoder = vae_model.encoder
    embedding.weight.requires_grad = False
    encoder.untrain()
    model = Classifier(rnn_hidden_dim=ENCODER_HIDDEN_SIZE,
                       mid_hidden_dim1=MID_HIDDEN_1, mid_hidden_dim2=MID_HIDDEN_2, class_number=2).cuda()
    optim = Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    for e in range(EPOCHS):
        for batch in range(len(training_data) // BATCH_SIZE):
            raw = [(training_data[batch * BATCH_SIZE + i].cuda(), training_data.get_tag(batch * BATCH_SIZE + i)) for i
                     in range(BATCH_SIZE)]
            raw.sort(key=lambda x: len(x[0]), reverse=True)
            lengths = [len(x[0]) for x in raw]
            input = Variable(torch.zeros(BATCH_SIZE, lengths[0], EMBEDDING_SIZE).cuda(), requires_grad=False)
            for i in range(BATCH_SIZE):
                input[i, :lengths[i]] = embedding(raw[i][0])
            target = Variable(torch.LongTensor([x[1] for x in raw]).cuda(), requires_grad=False)
            pack = nn.utils.rnn.pack_padded_sequence(input, lengths=lengths, batch_first=True)
            _, hidden, _ = encoder(pack, lengths=lengths, variation=variation)
            scores = model(hidden)
            loss = loss_fn(scores, target)
            optim.zero_grad()
            loss.backward()
            optim.step()
            if batch % 10 == 0:
                print("Epoch {}, Batch {}, Loss {}".format(e, batch, loss.data[0]))
        print("Saving checkpoints...")
        torch.save(
            {"Epoch": e,
             "Loss": loss.data[0],
             "model_state_dict": model.state_dict(),
             "optimizer_state_dict": optim.state_dict()}, MODEL_FILE_PATH)
