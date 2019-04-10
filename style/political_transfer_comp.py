import sys

sys.path.append('/dscrhome/hw186/pytorch_toy')
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
import torch.nn.functional as F

from vae.main_module.variational_autoencoder_comp import VAE
from vae.data_loader.data_loader_comp import VAEData
from classifier.political_classifier_comp import Classifier

if __name__ == "__main__":
    BATCH_SIZE = 30
    MAX_SEQ_LEN = 50
    ENCODER_HIDDEN_SIZE = 300
    DECODER_HIDDEN_SIZE = 300
    LEARNING_RATE = 1e-3
    EPOCHS = 300
    EMBEDDING_SIZE = 500
    BEAM_WIDTH = 3
    RECONSTRUCTION_COEFFICIENT = 0.02
    VOCAB = "../data/vocab.txt"
    TRAINING = "../data/mixed_train.txt"
    TESTING = "../data/democratic_only.test.en"
    PRETRAINED_MODEL_FILE_PATH = "../vae/model/checkpoint.pt"
    MODEL_FILE_PATH = "model/republican_style.pt"
    CLASSIFIER_MODEL_FILE_PATH = "../classifier/model/checkpoint.pt"
    pretrained = True
    variation = False
    DESIRED_STYLE = 1
    HIDDEN_DIM = 50
    MID_HIDDEN_1 = 50
    MID_HIDDEN_2 = 10

    training_dataset = VAEData(filepath=TRAINING, vocab_file=VOCAB, max_seq_len=MAX_SEQ_LEN, vocab_file_offset=1,
                               data_file_offset=1)
    model = VAE(embed=EMBEDDING_SIZE, encoder_hidden=ENCODER_HIDDEN_SIZE, decoder_hidden=DECODER_HIDDEN_SIZE,
                vocabulary=training_dataset.get_vocab_size()).cuda()
    model.embedding.weight.requires_grad = False
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    if pretrained:
        model.load_state_dict(torch.load(PRETRAINED_MODEL_FILE_PATH)["model_state_dict"])
    classification_loss_fn = nn.CrossEntropyLoss()
    reconstruction_loss_fn = nn.MSELoss(size_average=False)
    classifier = Classifier(vocab_size=training_dataset.get_vocab_size(), rnn_hidden_dim=HIDDEN_DIM,
                            mid_hidden_dim1=MID_HIDDEN_1, mid_hidden_dim2=MID_HIDDEN_2, class_number=2).cuda()
    classifier.load_state_dict(torch.load(CLASSIFIER_MODEL_FILE_PATH)["model_state_dict"])
    classifier.untrain()
    for epoch in range(EPOCHS):
        for batch in range(len(training_dataset) // BATCH_SIZE):
            input = [Variable(training_dataset[batch * BATCH_SIZE + i].cuda()) for i in range(BATCH_SIZE)]
            input.sort(key=lambda seq: len(seq), reverse=True)
            lengths = [len(seq) for seq in input]
            out, kl_loss = model(input, lengths, teacher_forcing_ratio=0.9, variation=variation)
            # out: [batch, max_seq_len, vocab_size]
            packed_out = nn.utils.rnn.pack_padded_sequence(F.softmax(out, dim=2), lengths, batch_first=True)
            target = torch.zeros(BATCH_SIZE).type(torch.LongTensor).cuda()
            target.fill_(DESIRED_STYLE)
            pred = classifier(packed_out)
            style_loss = classification_loss_fn(pred, Variable(target, requires_grad=False)) / BATCH_SIZE
            reconstruction_loss = Variable(torch.zeros(1).cuda())
            for i in range(1, BATCH_SIZE):
                reconstructed = torch.matmul(F.softmax(out[i, :lengths[i]], dim=1), model.embedding.weight)
                reconstruction_loss += reconstruction_loss_fn(reconstructed, model.embedding(input[i]))
            reconstruction_loss = reconstruction_loss / BATCH_SIZE
            total_loss = RECONSTRUCTION_COEFFICIENT * reconstruction_loss + (
                    1 - RECONSTRUCTION_COEFFICIENT) * style_loss + kl_loss
            optim.zero_grad()
            total_loss.backward()
            optim.step()
            if batch % 10 == 0:
                print(
                    "Epoch {}, Batch {}, KL Loss {}, Reconstruction Loss {}, Style Loss {}, Total Loss {}".format(epoch,
                                                                                                                  batch,
                                                                                                                  kl_loss.data[
                                                                                                                      0],
                                                                                                                  reconstruction_loss.data[
                                                                                                                      0],
                                                                                                                  style_loss.data[
                                                                                                                      0],
                                                                                                                  total_loss.data[
                                                                                                                      0]))
            if batch % 1000 == 0:
                print("Saving checkpoints...")
                torch.save(
                    {"Epoch": epoch, "KL Loss": kl_loss.data[0], "Reconstruction Loss": reconstruction_loss.data[0],
                     "Style Loss": style_loss.data[0],
                     "Total Loss": total_loss.data[0],
                     "model_state_dict": model.state_dict(),
                     "optimizer_state_dict": optim.state_dict()}, MODEL_FILE_PATH)
