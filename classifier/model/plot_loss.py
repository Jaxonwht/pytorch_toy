import matplotlib.pyplot as plt

with open("slurm-51264270.out") as f:
    i = 0
    for line in f:
        if line.startswith("S"):
            continue
        plt.scatter(i, float(line.split(" ")[5]), marker='x', color='r')
        i += 1
plt.show()
