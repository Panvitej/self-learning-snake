import csv
import os


class Logger:
    def __init__(self, path):
        self.path = path

        os.makedirs(os.path.dirname(path), exist_ok=True)

        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "score", "epsilon"])

    def log(self, episode, score, epsilon):
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode, score, epsilon])
