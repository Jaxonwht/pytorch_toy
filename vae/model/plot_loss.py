import matplotlib.pyplot as plt

with open("slurm-51164028.out") as f:
    i = 0
    for line in f:
        plt.scatter(i, float(line.split(" ")[9][:-1]))
        i += 1
plt.show()
