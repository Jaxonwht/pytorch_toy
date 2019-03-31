import sys

sys.path.append('/dscrhome/hw186/pytorch_toy')
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
from torch.nn.modules.linear import Linear
from torch.nn.modules.sparse import Embedding

from vae.data_loader.data_loader_comp import VAEData
from vae.main_module.decoder_unit_comp import Decoder
from vae.main_module.encoder_unit_comp import Encoder


class VAE(nn.Module):
    def __init__(self, embed, encoder_hidden, decoder_hidden, embedding_weights=None, vocabulary=None):
        super().__init__()
        if vocabulary:
            embedding = Embedding(vocabulary, embed)
        else:
            embedding = Embedding(embedding_weights.size()[0], embed)
            embedding.weight.data = embedding_weights
        self.encoder = Encoder(embedding_layer=embedding, hidden=encoder_hidden)
        self.translator = Linear(in_features=encoder_hidden * 2, out_features=decoder_hidden)
        self.translator_activation = nn.LeakyReLU()
        self.decoder = Decoder(encoder_hidden=encoder_hidden, decoder_hidden=decoder_hidden, embedding_layer=embedding)

    def forward(self, x, lengths, teacher_forcing_ratio=0.5, variation=True):
        '''
        :param x: list of tensors, len(list) = batch, each tensor is [variable_seq_len]
        :param lengths: [batch]
        '''
        encoder_outs, encoder_hidden, kl_loss = self.encoder(x, lengths, variation=variation)
        decoder_hidden = self.translator_activation(self.translator(encoder_hidden))
        out = self.decoder(x, decoder_hidden, encoder_outs, lengths, teacher_forcing_ratio=teacher_forcing_ratio)
        return out, kl_loss


if __name__ == "__main__":
    BATCH_SIZE = 50
    MAX_SEQ_LEN = 50
    ENCODER_HIDDEN_SIZE = 200
    DECODER_HIDDEN_SIZE = 200
    LEARNING_RATE = 1e-2
    EPOCHS = 300
    EMBEDDING_SIZE = 100
    VOCAB = "../../data/classtrain.txt"
    TRAINING = "../../data/mixed_train.txt"
    WORD2VEC_WEIGHT = "../../word2vec/model/model_state_dict.pt"
    MODEL_FILE_PATH = "../model/checkpoint.pt"
    pretrained = False
    variation = False

    training_dataset = VAEData(filepath=TRAINING, vocab_data_file=VOCAB, max_seq_len=MAX_SEQ_LEN)
    model = VAE(embed=EMBEDDING_SIZE, encoder_hidden=ENCODER_HIDDEN_SIZE, decoder_hidden=DECODER_HIDDEN_SIZE,
                vocabulary=training_dataset.get_vocab_size()).cuda()
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    if pretrained:
        model.load_state_dict(torch.load(MODEL_FILE_PATH)["model_state_dict"])
        # optim.load_state_dict(torch.load(MODEL_FILE_PATH)["optimizer_state_dict"])
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1, size_average=False)
    for epoch in range(EPOCHS):
        for batch in range(len(training_dataset) // BATCH_SIZE):
            input = [Variable(training_dataset[batch * BATCH_SIZE + i].cuda()) for i in range(BATCH_SIZE)]
            input.sort(key=lambda seq: len(seq), reverse=True)
            lengths = [len(seq) for seq in input]
            out, kl_loss = model(input, lengths, teacher_forcing_ratio=0.9, variation=variation)
            padded_input = Variable(torch.zeros(len(lengths), lengths[0]).type(torch.LongTensor).cuda())
            # padded_input = [batch, max_seq_len]
            padded_input.fill_(-1)
            for batch_index in range(len(lengths)):
                padded_input[batch_index, :lengths[batch_index]] = input[batch_index]
            out = out.permute(0, 2, 1)
            # out: [batch, max_seq_len, vocab_size] -> [batch, vocab_size, max_seq_len]
            reconstruction_loss = Variable(torch.zeros(1).cuda())
            for token_index in range(1, lengths[0]):
                reconstruction_loss += loss_fn(out[:, :, token_index], padded_input[:, token_index])
            total_loss = reconstruction_loss
            optim.zero_grad()
            total_loss.backward()
            optim.step()
            if batch % 10 == 0:
                print("Epoch {}, Batch {}, KL Loss {}, Reconstruction Loss {}, Total Loss {}".format(epoch, batch,
                                                                                                     kl_loss.data[0],
                                                                                                     reconstruction_loss.data[
                                                                                                         0],
                                                                                                     total_loss.data[
                                                                                                         0]))
        print("Saving checkpoints...")
        torch.save(
            {"Epoch": epoch, "KL Loss": kl_loss.data[0], "Reconstruction Loss": reconstruction_loss.data[0],
             "Total Loss": total_loss.data[0],
             "model_state_dict": model.state_dict(),
             "optimizer_state_dict": optim.state_dict()}, MODEL_FILE_PATH)
