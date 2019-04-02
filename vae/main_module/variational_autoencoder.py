import torch.nn as nn
import torch.optim
from torch.nn.modules.linear import Linear
from torch.nn.modules.sparse import Embedding

from vae.data_loader.data_loader import VAEData
from vae.main_module.decoder_unit import Decoder
from vae.main_module.encoder_unit import Encoder
from word2vec.main_module.word_to_vec import EmailDataset


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
        self.decoder = Decoder(encoder_hidden=encoder_hidden, decoder_hidden=decoder_hidden, embedding_layer=embedding,
                               device=device)

    def forward(self, x, lengths, teacher_forcing_ratio, variation):
        '''
        :param x: list of tensors, len(list) = batch, each tensor is [variable_seq_len]
        :param lengths: [batch]
        '''
        encoder_outs, encoder_hidden, kl_loss = self.encoder(x, lengths, variation=variation)
        decoder_hidden = self.translator_activation(self.translator(encoder_hidden))
        out = self.decoder(x, decoder_hidden, encoder_outs, lengths, teacher_forcing_ratio=teacher_forcing_ratio)
        return out, kl_loss

    def inference(self, input, beam_width, variation):
        '''
        :param input: [seq_len]
        :param beam_width: scalar
        :param variation: boolean
        :return: a tensor of variable length
        '''
        with torch.no_grad():
            encoder_outs, encoder_hidden, _ = self.encoder([input], torch.tensor([len(input)]), variation=variation)
            decoder_hidden = self.translator_activation(self.translator(encoder_hidden))
            out = self.decoder.inference(initial_hidden=decoder_hidden, encoder_outs=encoder_outs,
                                         beam_width=beam_width, length=torch.tensor([len(input)]))
            return out


if __name__ == "__main__":
    if torch.cuda.is_available():
        my_device = torch.device("cuda")
    else:
        my_device = torch.device("cpu")
    BATCH_SIZE = 50
    MAX_SEQ_LEN = 50
    ENCODER_HIDDEN_SIZE = 300
    DECODER_HIDDEN_SIZE = 300
    LEARNING_RATE = 1e-3
    EPOCHS = 300
    EMBEDDING_SIZE = 500
    BEAM_WIDTH = 3
    VOCAB = "../../data/classtrain.txt"
    TRAINING = "../../data/mixed_train.txt"
    WORD2VEC_WEIGHT = "../../word2vec/model/model_state_dict.pt"
    TESTING = "../../data/democratic_only.test.en"
    PRETRAINED_MODEL_FILE_PATH = "../model/checkpoint.pt"
    MODEL_FILE_PATH = "../model/checkpoint_variation.pt"
    training = False
    pretrained = True
    variation = False

    if training:
        training_dataset = VAEData(filepath=TRAINING, vocab_data_file=VOCAB, max_seq_len=MAX_SEQ_LEN, offset=1)
        model = VAE(embed=EMBEDDING_SIZE, encoder_hidden=ENCODER_HIDDEN_SIZE, decoder_hidden=DECODER_HIDDEN_SIZE,
                    device=my_device, vocabulary=training_dataset.get_vocab_size()).to(my_device)
        optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        if pretrained:
            model.load_state_dict(torch.load(PRETRAINED_MODEL_FILE_PATH)["model_state_dict"])
            # optim.load_state_dict(torch.load(MODEL_FILE_PATH)["optimizer_state_dict"])
        loss_fn = nn.CrossEntropyLoss(ignore_index=-1, reduction="sum")
        for epoch in range(EPOCHS):
            for batch in range(len(training_dataset) // BATCH_SIZE):
                input = [training_dataset[batch * BATCH_SIZE + i].to(my_device) for i in range(BATCH_SIZE)]
                input.sort(key=lambda seq: len(seq), reverse=True)
                lengths = torch.tensor([len(seq) for seq in input]).to(my_device)
                out, kl_loss = model(input, lengths, teacher_forcing_ratio=0.9, variation=variation)
                padded_input = nn.utils.rnn.pad_sequence(input, batch_first=True, padding_value=-1).to(my_device)
                # padded_input = [batch, max_seq_len]
                out = out.permute(0, 2, 1)
                # out: [batch, max_seq_len, vocab_size] -> [batch, vocab_size, max_seq_len]
                reconstruction_loss = torch.zeros(1).to(my_device)
                for i in range(1, lengths[0]):
                    reconstruction_loss += loss_fn(out[:, :, i], padded_input[:, i])
                reconstruction_loss = reconstruction_loss / BATCH_SIZE
                # reconstruction_loss = torch.zeros(1, device=my_device)
                # for token_index in range(1, lengths[0]):
                #     reconstruction_loss += loss_fn(out[:, :, token_index], padded_input[:, token_index])
                total_loss = reconstruction_loss + kl_loss
                optim.zero_grad()
                total_loss.backward()
                optim.step()
                print("Epoch {}, Batch {}, KL Loss {}, Reconstruction Loss {}, Total Loss {}".format(epoch, batch,
                                                                                                     kl_loss.item(),
                                                                                                     reconstruction_loss.item(),
                                                                                                     total_loss.item()))
    else:
        vocab_dataset = EmailDataset(VOCAB, 0)
        testing_dataset = VAEData(filepath=TESTING, vocab_data_file=VOCAB, max_seq_len=MAX_SEQ_LEN, offset=0)
        model = VAE(embed=EMBEDDING_SIZE, encoder_hidden=ENCODER_HIDDEN_SIZE, decoder_hidden=DECODER_HIDDEN_SIZE,
                    device=my_device, vocabulary=testing_dataset.get_vocab_size()).to(my_device)
        if pretrained:
            model.load_state_dict(torch.load(PRETRAINED_MODEL_FILE_PATH)["model_state_dict"])
        input = [testing_dataset[i].to(my_device) for i in range(BATCH_SIZE)]
        for i in range(BATCH_SIZE):
            input = testing_dataset[i + 20].to(my_device)
            print("The original sequence is:")
            print([vocab_dataset.get_token(j) for j in input.tolist()])
            out = model.inference(input=input, beam_width=BEAM_WIDTH, variation=False)
            print("The translated sequence is:")
            print([vocab_dataset.get_token(j.item()) for j in out])
            print()
