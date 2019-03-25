mixed = open("mixed_train.txt", 'w')

with open("democratic_only.train.en") as d:
    with open("republican_only.train.en") as r:
        while d.readable() or r.readable():
            if d.readable():
                mixed.write(d.readline())
            if r.readable():
                mixed.write(r.readline())

mixed.close()