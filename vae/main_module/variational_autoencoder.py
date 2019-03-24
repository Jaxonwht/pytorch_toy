import torch.nn as nn
import torch.optim
from torch.nn.modules.linear import Linear
from vae.main_module.decoder_unit import Decoder
from torch.nn.modules.sparse import Embedding

from vae.data_loader.data_loader import VAEData
from vae.main_module.encoder_unit import Encoder


class VAE(nn.Module):
    def __init__(self, embed, encoder_hidden, decoder_hidden, device, embedding_weights=None, vocabulary=None):
        super().__init__()
        if vocabulary:
            embedding = Embedding(vocabulary, embed)
        else:
            embedding = Embedding.from_pretrained(embeddings=embedding_weights)
        self.encoder = Encoder(embedding_layer=embedding, hidden=encoder_hidden, device=device)
        self.translator = Linear(in_features=encoder_hidden * 2, out_features=decoder_hidden)
        self.translator_activation = nn.LeakyReLU()
        self.decoder = Decoder(encoder_hidden=encoder_hidden, decoder_hidden=decoder_hidden, embedding_layer=embedding, device=device)

    def forward(self, x, lengths):
        '''
        :param x: list of tensors, len(list) = batch, each tensor is [variable_seq_len]
        :param lengths: [batch]
        '''
        encoder_outs, encoder_hidden, kl_loss = self.encoder(x, lengths)
        decoder_hidden = self.translator_activation(self.translator(encoder_hidden))
        out = self.decoder(x, decoder_hidden, encoder_outs, lengths)
        return out, kl_loss


if __name__ == "__main__":
    if torch.cuda.is_available():
        my_device = torch.device("cuda")
    else:
        my_device = torch.device("cpu")
    BATCH_SIZE = 32
    MAX_SEQ_LEN = 50
    ENCODER_HIDDEN_SIZE = 200
    DECODER_HIDDEN_SIZE = 200
    LEARNING_RATE = 1e-3
    EPOCHS = 300
    EMBEDDING_SIZE = 500
    VOCAB = "../../data/classtrain.txt"
    TRAINING = "../../data/democratic_only.dev.en"
    WORD2VEC_WEIGHT = "../../word2vec/model/model_state_dict.pt"

    training_dataset = VAEData(filepath=TRAINING, vocab_data_file=VOCAB, max_seq_len=MAX_SEQ_LEN)
    model = VAE(embed=EMBEDDING_SIZE, encoder_hidden=ENCODER_HIDDEN_SIZE, decoder_hidden=DECODER_HIDDEN_SIZE, device=my_device, embedding_weights=torch.load(WORD2VEC_WEIGHT)["embed.weight"]).to(my_device)
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    total_loss = torch.zeros(1).to(my_device)
    for epoch in range(EPOCHS):
        for batch in range(len(training_dataset) // BATCH_SIZE):
            input = [training_dataset[batch * BATCH_SIZE + i].to(my_device) for i in range(BATCH_SIZE)]
            input.sort(key=lambda seq: len(seq), reverse=True)
            lengths = torch.tensor([len(seq) for seq in input]).to(my_device)
            out, kl_loss = model(input, lengths)
            padded_input = nn.utils.rnn.pad_sequence(input, batch_first=True, padding_value=-1).to(my_device)
            # padded_input = [batch, max_seq_len]
            out = out.permute(0, 2, 1)
            # out: [batch, max_seq_len, vocab_size] -> [batch, vocab_size, max_seq_len]
            total_loss = kl_loss + loss_fn(out[:, :, 1:], padded_input[:, 1:])
            optim.zero_grad()
            total_loss.backward()
            optim.step()
            print("Epoch {}, Batch {}, loss {}".format(epoch, batch, total_loss.item()))