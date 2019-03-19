import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from word2vec.word_to_vec import Word2Vec

input = torch.randint(0, 5, (10,))
target = torch.randint(0, 5, (10,))

model = Word2Vec(5, 2)
optimizer = Adam(model.parameters(), lr=4)
loss_fn = CrossEntropyLoss()

for epoch in range(10000):
    pred = model(input)
    loss = loss_fn(pred, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("Epoch: {}, loss: {}".format(epoch, loss.item()))

print(input)
print("====")
print(target)
print("====")
model.eval()
print(torch.nn.Softmax()(model(input)))
