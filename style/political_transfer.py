import sys

sys.path.append("/workspace/pytorch_toy/")
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

from classifier.political_classifier import Classifier
from vae.data_loader.data_loader import VAEData
from vae.main_module.variational_autoencoder import VAE

if __name__ == "__main__":
    if torch.cuda.is_available():
        my_device = torch.device("cuda")
    else:
        my_device = torch.device("cpu")
    ATTENTION = False
    BATCH_SIZE = 50
    MAX_SEQ_LEN = 50
    ENCODER_HIDDEN_SIZE = 300
    DECODER_HIDDEN_SIZE = 300
    LEARNING_RATE = 1e-4
    EPOCHS = 300
    EMBEDDING_SIZE = 300
    BEAM_WIDTH = 3
    VOCAB = "../data/vocab.txt"
    TRAINING = "../data/mixed_train.txt"
    TESTING = "../data/democratic_only.test.en"
    if ATTENTION:
        VAE_MODEL_FILE_PATH = "../vae/model/checkpoint_attention.pt"
        PRETRAINED_MODEL_FILE_PATH = "model/republican_style_attention.pt"
        MODEL_FILE_PATH = "model/republican_style_attention.pt"
    else:
        VAE_MODEL_FILE_PATH = "../vae/model/checkpoint.pt"
        PRETRAINED_MODEL_FILE_PATH = "model/republican_style.pt"
        MODEL_FILE_PATH = "model/republican_style.pt"
    CLASSIFIER_MODEL_FILE_PATH = "../classifier/model/checkpoint.pt"
    training = False
    pretrained = True
    variation = True
    DESIRED_STYLE = 1
    HIDDEN_DIM = 50
    MID_HIDDEN_1 = 50
    MID_HIDDEN_2 = 10

    if training:
        training_dataset = VAEData(filepath=TRAINING, vocab_file=VOCAB, max_seq_len=MAX_SEQ_LEN, vocab_file_offset=1,
                                   data_file_offset=1, min_freq=2)
        model = VAE(embed=EMBEDDING_SIZE, encoder_hidden=ENCODER_HIDDEN_SIZE, decoder_hidden=DECODER_HIDDEN_SIZE,
                    device=my_device, vocabulary=training_dataset.get_vocab_size(), attention=ATTENTION).to(my_device)
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
        if pretrained:
            model.load_state_dict(torch.load(PRETRAINED_MODEL_FILE_PATH)["model_state_dict"])
        else:
            model.load_state_dict(torch.load(VAE_MODEL_FILE_PATH)["model_state_dict"])
        model.embedding.weight.requires_grad = False
        model.encoder.untrain()
        classification_loss_fn = nn.CrossEntropyLoss()
        reconstruction_loss_fn = nn.MSELoss(reduction="sum")
        classifier = Classifier(vocab_size=training_dataset.get_vocab_size(), rnn_hidden_dim=HIDDEN_DIM,
                                mid_hidden_dim1=MID_HIDDEN_1, mid_hidden_dim2=MID_HIDDEN_2, class_number=2).to(
            my_device)
        classifier.load_state_dict(torch.load(CLASSIFIER_MODEL_FILE_PATH)["model_state_dict"])
        classifier.untrain()
        for epoch in range(EPOCHS):
            for batch in range(len(training_dataset) // BATCH_SIZE):
                input = [training_dataset[batch * BATCH_SIZE + i].to(my_device) for i in range(BATCH_SIZE)]
                input.sort(key=lambda seq: len(seq), reverse=True)
                lengths = torch.tensor([len(x) for x in input])
                out, _, before_hidden = model(input, lengths, teacher_forcing_ratio=0.9, variation=variation)
                # out = [batch, max_seq_len, vocab_size]
                # before_hidden = [batch, encoder_hidden]
                packed_out = torch.nn.utils.rnn.pack_padded_sequence(input=F.softmax(out, dim=2), lengths=lengths,
                                                                     batch_first=True)
                pred = classifier(packed_out)
                # pred = [batch, class_number]
                target = torch.zeros(BATCH_SIZE, dtype=torch.long, device=my_device)
                target.fill_(DESIRED_STYLE)
                style_loss = classification_loss_fn(pred, target)
                # padded_input = [batch, max_seq_len]
                reconstruction_loss = torch.zeros(1, device=my_device)
                _, after_hidden, _ = model.encoder(packed_out, lengths, variation=variation, attention=ATTENTION)
                reconstruction_loss = reconstruction_loss_fn(before_hidden, after_hidden) / BATCH_SIZE
                
                # old reconstruction loss
                # for i in range(BATCH_SIZE):
                # reconstruction_loss += reconstruction_loss_fn(
                # torch.matmul(F.softmax(out[i, :lengths[i]], dim=1), model.embedding.weight),
                # model.embedding(input[i]))
                
                total_loss = reconstruction_loss / 20 + style_loss / 2
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
                if batch % 1000 == 0:
                    print("Saving checkpoints...")
                    torch.save(
                        {"Epoch": epoch, "KL Loss": kl_loss.item(), "Reconstruction Loss": reconstruction_loss.item(),
                         "Style Loss": style_loss.item(),
                         "Total Loss": total_loss.item(),
                         "model_state_dict": model.state_dict(),
                         "optimizer_state_dict": optim.state_dict()}, MODEL_FILE_PATH)
    else:
        testing_dataset = VAEData(filepath=TESTING, vocab_file=VOCAB, max_seq_len=MAX_SEQ_LEN, vocab_file_offset=1,
                                  data_file_offset=0, min_freq=2)
        model = VAE(embed=EMBEDDING_SIZE, encoder_hidden=ENCODER_HIDDEN_SIZE, decoder_hidden=DECODER_HIDDEN_SIZE,
                    device=my_device, vocabulary=testing_dataset.get_vocab_size(), attention=ATTENTION).to(my_device)
        if pretrained:
            model.load_state_dict(torch.load(MODEL_FILE_PATH)["model_state_dict"])
        for i in range(10):
            input = testing_dataset[i].to(my_device)
            print("The original sequence is:")
            print([testing_dataset.get_token(j) for j in input.tolist()])
            # with torch.no_grad():
            out = model.inference(input=input, beam_width=BEAM_WIDTH, variation=False, max_seq_len=MAX_SEQ_LEN)
            print("The translated sequence is:")
            print([testing_dataset.get_token(j.item()) for j in out])
            print()
