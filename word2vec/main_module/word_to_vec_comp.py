import sys

sys.path.append('/dscrhome/hw186/pytorch_toy')
import torch
import torch.nn as nn
import torch.optim as optim
from word2vec.data_loader.data_loader_comp import EmailDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable

FILE_PATH = "../../data/classtrain.txt"
OUT_MODEL_STATE_DICT = "../model/model_state_dict.pt"
OUT_OPTIM_STATE_DICT = "../model/optim_state_dict.pt"
if torch.cuda.is_available():
    my_device = torch.cuda.set_device(0)
else:
    my_device = torch.cuda.set_device(-1)
CONTEXT_SIZE = 2
BATCH_SIZE = 32
HIDDEN_SIZE = 500
LEARNING_RATE = 1e-4
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
        return self.output(self.embed(Variable(x)))


model = Word2Vec(vocab_size, HIDDEN_SIZE).cuda()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for e in range(EPOCHS):
    for x, y in enumerate(my_data_loader):
        input_vector = y[:, 0].cuda()
        target_pred = model(input_vector).cuda()
        target_vector = Variable(y[:, 1].cuda())
        loss = loss_fn(target_pred, target_vector).cuda()
        if x % 1000 == 0:
            print("Epoch: {0}, Batch: {1}, loss: {2}".format(e, x, loss.data[0]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), OUT_MODEL_STATE_DICT)
    torch.save(optimizer.state_dict(), OUT_OPTIM_STATE_DICT)

torch.save(model.state_dict(), OUT_MODEL_STATE_DICT)
torch.save(optimizer.state_dict(), OUT_OPTIM_STATE_DICT)
# model.load_state_dict(torch.load(out_model_state_dict))
# optimizer.load_state_dict(torch.load(out_optim_state_dict))
# model.eval()

# plot all the points
# with torch.no_grad():
#     coordinates_high = model.embed.weight.data[0 : 200].cpu().numpy()
#     tsne_model = TSNE(n_components=2, init='pca')
#     coordinates_low = tsne_model.fit_transform(coordinates_high)
#     for i, triple in enumerate(coordinates_low):
#         plt.scatter(triple[0], triple[1], marker='x')
#         plt.annotate(email_data.get_token(i), xy=(triple[0], triple[1]))
#     plt.show()
