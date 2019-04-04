import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from vae.data_loader.data_loader import VAEData
from vae.main_module.variational_autoencoder import VAE


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
    if torch.cuda.is_available():
        my_device = torch.device("cuda")
    else:
        my_device = torch.device("cpu")
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
    training = True
    variation = False

    if training:
        training_data = VAEData(filepath=TRAINING, vocab_file=VOCAB, max_seq_len=MAX_SEQ_LEN, data_file_offset=1,
                                vocab_file_offset=1)
        vae_model = VAE(embed=EMBEDDING_SIZE, encoder_hidden=ENCODER_HIDDEN_SIZE,
                                decoder_hidden=DECODER_HIDDEN_SIZE,
                                device=my_device, vocabulary=training_data.get_vocab_size()).to(my_device)
        vae_model.load_state_dict(torch.load(VAE_MODEL_FILE_PATH)["model_state_dict"], strict=False)
        embedding = vae_model.embedding
        encoder = vae_model.encoder
        embedding.weight.requires_grad = False
        encoder.untrain()
        model = Classifier(rnn_hidden_dim=ENCODER_HIDDEN_SIZE,
                           mid_hidden_dim1=MID_HIDDEN_1, mid_hidden_dim2=MID_HIDDEN_2, class_number=2).to(my_device)
        optim = Adam(model.parameters(), lr=LEARNING_RATE)
        if pretrained:
            model.load_state_dict(torch.load(PRETRAINED_MODEL_FILE_PATH)["model_state_dict"])
            optim.load_state_dict(torch.load(PRETRAINED_MODEL_FILE_PATH)["optimizer_state_dict"])
        loss_fn = nn.CrossEntropyLoss()
        for e in range(EPOCHS):
            for batch in range(len(training_data) // BATCH_SIZE):
                raw = [(training_data[batch * BATCH_SIZE + i].to(my_device), training_data.get_tag(batch * BATCH_SIZE + i)) for i in range(BATCH_SIZE)]
                raw.sort(key=lambda x: len(x[0]), reverse=True)
                lengths = torch.tensor([len(x[0]) for x in raw], device=my_device)
                input = [embedding(raw[i][0]) for i in range(BATCH_SIZE)]
                target = torch.tensor([x[1] for x in raw], device=my_device, requires_grad=False)
                pack = nn.utils.rnn.pack_sequence(input).to(my_device)
                _, hidden, _ = encoder(pack, lengths=lengths, variation=variation)
                scores = model(hidden)
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
        vae_model = VAE(embed=EMBEDDING_SIZE, encoder_hidden=ENCODER_HIDDEN_SIZE,
                        decoder_hidden=DECODER_HIDDEN_SIZE,
                        device=my_device, vocabulary=training_data.get_vocab_size()).to(my_device)
        vae_model.load_state_dict(torch.load(VAE_MODEL_FILE_PATH)["model_state_dict"], strict=False)
        embedding = vae_model.embedding
        encoder = vae_model.encoder
        embedding.weight.requires_grad = False
        encoder.untrain()
        model = Classifier(rnn_hidden_dim=ENCODER_HIDDEN_SIZE,
                           mid_hidden_dim1=MID_HIDDEN_1, mid_hidden_dim2=MID_HIDDEN_2, class_number=2).to(my_device)
        model.load_state_dict(torch.load(MODEL_FILE_PATH)["model_state_dict"])
        model.eval()
        for e in range(EPOCHS):
            for batch in range(len(training_data) // BATCH_SIZE):
                raw = [
                    (training_data[batch * BATCH_SIZE + i].to(my_device), training_data.get_tag(batch * BATCH_SIZE + i))
                    for i in range(BATCH_SIZE)]
                raw.sort(key=lambda x: len(x[0]), reverse=True)
                lengths = torch.tensor([len(x[0]) for x in raw], device=my_device)
                input = torch.zeros(BATCH_SIZE, lengths[0].item(), EMBEDDING_SIZE, device=my_device)
                target = torch.tensor([x[1] for x in raw], device=my_device)
                for i in range(BATCH_SIZE):
                    input[i, :lengths[i]] = embedding(raw[i][0])
                pack = nn.utils.rnn.pack_sequence(input).to(my_device)
                _, hidden, _ = encoder(pack, lengths=lengths, variation=variation)
                scores = model(hidden)
                print(F.softmax(scores))
