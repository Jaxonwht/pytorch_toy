vocab = open("vocab.txt", 'w')

d_train = open("democratic_only.train.en", 'r')
r_train = open("republican_only.train.en", 'r')
d_dev = open("democratic_only.dev.en", 'r')
r_dev = open("republican_only.dev.en", 'r')
d_test = open("democratic_only.test.en", 'r')
r_test = open("republican_only.test.en", 'r')

for line in d_train:
    if not line.isspace():
        vocab.write(line)
for line in r_train:
    if not line.isspace():
        vocab.write(line)
for line in d_dev:
    if not line.isspace():
        vocab.write(line)
for line in r_dev:
    if not line.isspace():
        vocab.write(line)
for line in d_test:
    if not line.isspace():
        vocab.write(line)
for line in r_test:
    if not line.isspace():
        vocab.write(line)

vocab.close()
d_train.close()
r_train.close()
d_dev.close()
r_dev.close()
d_test.close()
r_test.close()
