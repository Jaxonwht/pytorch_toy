import random

NUMBER_EXPERIMENTS = 10 ** 5
p = 0.7

counts = 0
for i in range(NUMBER_EXPERIMENTS):
    head = 0
    tail = 1
    count = 0
    while head < tail:
        toss = random.random()
        if toss < 0.7:
            head += 1
        else:
            tail += 1
        count += 1
    counts += count
print(counts / NUMBER_EXPERIMENTS)
