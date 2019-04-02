import torch.nn as nn
import torch.optim

from classifier.political_classifier import Classifier
from vae.data_loader.data_loader import VAEData
from vae.main_module.variational_autoencoder import VAE

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
    RECONSTRUCTION_COEFFICIENT = 0.7
    VOCAB = "../data/classtrain.txt"
    TRAINING = "../data/mixed_train.txt"
    TESTING = "../data/democratic_only.test.en"
    PRETRAINED_MODEL_FILE_PATH = "../vae/model/checkpoint.pt"
    MODEL_FILE_PATH = "model/checkpoint.pt"
    CLASSIFIER_MODEL_FILE_PATH = "../classifier/model/checkpoint.pt"
    training = True
    pretrained = True
    variation = False
    DESIRED_STYLE = 1
    CLASSIFIER_RNN_HIDDEN_DIM = 150
    CLASSIFIER_MID_HIDDEN = 50
    CLASSIFIER_RNN_LAYERS = 2

    if training:
        training_dataset = VAEData(filepath=TRAINING, vocab_file=VOCAB, max_seq_len=MAX_SEQ_LEN, vocab_file_offset=1,
                                   data_file_offset=1)
        model = VAE(embed=EMBEDDING_SIZE, encoder_hidden=ENCODER_HIDDEN_SIZE, decoder_hidden=DECODER_HIDDEN_SIZE,
                    device=my_device, vocabulary=training_dataset.get_vocab_size()).to(my_device)
        optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        if pretrained:
            model.load_state_dict(torch.load(PRETRAINED_MODEL_FILE_PATH)["model_state_dict"])
            # optim.load_state_dict(torch.load(MODEL_FILE_PATH)["optimizer_state_dict"])
        loss_fn = nn.CrossEntropyLoss(ignore_index=-1, reduction="sum")
        classifier = Classifier(vocab_size=training_dataset.get_vocab_size(), rnn_hidden_dim=CLASSIFIER_RNN_HIDDEN_DIM,
                                rnn_layers=CLASSIFIER_RNN_LAYERS, mid_hidden_dim=CLASSIFIER_MID_HIDDEN,
                                class_number=2).to(my_device)
        classifier.load_state_dict(torch.load(CLASSIFIER_MODEL_FILE_PATH)["model_state_dict"])
        classifier.untrain()
        for epoch in range(EPOCHS):
            for batch in range(len(training_dataset) // BATCH_SIZE):
                input = [training_dataset[batch * BATCH_SIZE + i].to(my_device) for i in range(BATCH_SIZE)]
                input.sort(key=lambda seq: len(seq), reverse=True)
                lengths = torch.tensor([len(x) for x in input])
                out, kl_loss = model(input, lengths, teacher_forcing_ratio=0.9, variation=variation)
                # out = [batch, max_seq_len, vocab_size]
                packed_out = torch.nn.utils.rnn.pack_padded_sequence(input=out, lengths=lengths, batch_first=True)
                pred = classifier(packed_out)
                # pred = [batch, class_number]
                target = torch.zeros(BATCH_SIZE, dtype=torch.long, device=my_device)
                target.fill_(DESIRED_STYLE)
                style_loss = loss_fn(pred, target) / BATCH_SIZE
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
                total_loss = RECONSTRUCTION_COEFFICIENT * reconstruction_loss + (
                            1 - RECONSTRUCTION_COEFFICIENT) * style_loss + kl_loss
                optim.zero_grad()
                total_loss.backward()
                optim.step()
                print(
                    "Epoch {}, Batch {}, KL Loss {}, Reconstruction Loss {}, Style Loss {}, Total Loss {}".format(epoch,
                                                                                                                  batch,
                                                                                                                  kl_loss.item(),
                                                                                                                  reconstruction_loss.item(),
                                                                                                                  style_loss.item(),
                                                                                                                  total_loss.item()))
    else:
        testing_dataset = VAEData(filepath=TESTING, vocab_file=VOCAB, max_seq_len=MAX_SEQ_LEN, vocab_file_offset=1,
                                  data_file_offset=0)
        model = VAE(embed=EMBEDDING_SIZE, encoder_hidden=ENCODER_HIDDEN_SIZE, decoder_hidden=DECODER_HIDDEN_SIZE,
                    device=my_device, vocabulary=testing_dataset.get_vocab_size()).to(my_device)
        if pretrained:
            model.load_state_dict(torch.load(PRETRAINED_MODEL_FILE_PATH)["model_state_dict"])
        input = [testing_dataset[i].to(my_device) for i in range(BATCH_SIZE)]
        for i in range(BATCH_SIZE):
            input = testing_dataset[i + 20].to(my_device)
            print("The original sequence is:")
            print([testing_dataset.get_token(j) for j in input.tolist()])
            out = model.inference(input=input, beam_width=BEAM_WIDTH, variation=False)
            print("The translated sequence is:")
            print([testing_dataset.get_token(j.item()) for j in out])
            print()
