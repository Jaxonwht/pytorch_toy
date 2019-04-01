import matplotlib.pyplot as plt

with open("slurm-51200077.out") as f:
    i = 0
    for line in f:
        if line.startswith("S"):
            continue
        plt.scatter(i, float(line.split(" ")[9][:-1]), marker='x', color='r')
        i += 1
plt.show()
