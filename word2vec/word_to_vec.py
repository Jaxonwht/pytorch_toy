import torch
import torch.nn as nn
import torch.optim as optim
from word2vec.data_loader import EmailDataset
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


FILE_PATH = "../data/democratic5000.train"
OUT_MODEL_STATE_DICT = "model/model_state_dict.pt"
OUT_OPTIM_STATE_DICT = "model/optim_state_dict.pt"
if torch.cuda.is_available():
    gpu = torch.device("cuda")
else:
    gpu = torch.device("cpu")
CONTEXT_SIZE = 2
BATCH_SIZE = 32
HIDDEN_SIZE = 500
LEARNING_RATE = 1e-3
EPOCHS = 100

email_data = EmailDataset(FILE_PATH, CONTEXT_SIZE)
vocab_size = email_data.get_number_of_tokens()
my_data_loader = DataLoader(email_data, shuffle=True, batch_size=BATCH_SIZE)


class Word2Vec(nn.Module):
    def __init__(self, D, H):
        super().__init__()
        self.embed = nn.Embedding(D, H)
        self.output = nn.Linear(H, D)

    def forward(self, x):
        return self.output(self.embed(x))


model = Word2Vec(vocab_size, HIDDEN_SIZE).to(gpu)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for e in range(EPOCHS):
    for x, y in enumerate(my_data_loader):
        input_vector = y[:, 0].to(gpu)
        print(input_vector.size())
        target_pred = model(input_vector)
        print(target_pred.size())
        target_vector = y[:, 1].to(gpu)
        loss = loss_fn(target_pred, target_vector)
        if x % 1000 == 0:
            print("Epoch: {0}, Batch: {1}, loss: {2}".format(e, x, loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), OUT_MODEL_STATE_DICT)
torch.save(optimizer.state_dict(), OUT_OPTIM_STATE_DICT)
# model.load_state_dict(torch.load(out_model_state_dict))
# optimizer.load_state_dict(torch.load(out_optim_state_dict))
# model.eval()

# plot all the points
with torch.no_grad():
    data = model.weight.cpu().numpy()
    X_embedded = TSNE(n_components=2).fit_transform(data)
    plt.plot(X_embedded, ".")
    plt.show()
