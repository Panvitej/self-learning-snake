import csv
import matplotlib.pyplot as plt
from config import LOG_PATH


def plot():
    episodes = []
    scores = []

    with open(LOG_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            scores.append(float(row["score"]))

    plt.plot(episodes, scores)
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("Training Performance")
    plt.grid()

    plt.savefig("results/scores.png")
    plt.show()


if __name__ == "__main__":
    plot()
