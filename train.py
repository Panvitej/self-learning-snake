import os
import random
import numpy as np
import torch
import time

from snake_env import SnakeEnv
from agent import Agent
from config import EPISODES, MODEL_PATH, LOG_PATH, SEED
from utils.logger import Logger


# ---------------------- Core Optimization Utilities ----------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs():
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)


# ---------------------- Optimized Trainer ----------------------

class Trainer:
    def __init__(self):
        self.env = SnakeEnv(render=False)
        self.agent = Agent()
        self.logger = Logger(LOG_PATH)

        self.best_score = 0
        self.best_avg = 0

        self.scores = []
        self.start_time = time.time()

        # adaptive control
        self.no_improve_counter = 0

    # ---------------------- Training Loop ----------------------

    def train(self):
        for ep in range(EPISODES):
            state = self.env.reset()
            done = False

            score = 0
            steps = 0

            while not done:
                action = self.agent.act(state)
                next_state, reward, done = self.env.step(action)

                self.agent.remember(state, action, reward, next_state, done)

                # train multiple times per step → faster learning
                self.agent.train_step()
                if steps % 2 == 0:
                    self.agent.train_step()

                state = next_state
                score = self.env.score
                steps += 1

            self._post_episode(ep, score, steps)

        self._finalize()

    # ---------------------- Optimization Logic ----------------------

    def _post_episode(self, ep, score, steps):
        self.scores.append(score)

        avg = self._moving_avg(50)

        # decay epsilon slower if not learning
        if avg <= self.best_avg:
            self.no_improve_counter += 1
        else:
            self.no_improve_counter = 0
            self.best_avg = avg

        self._adaptive_decay()

        self.logger.log(ep, score, self.agent.epsilon)

        self._save_best(score)

        self._print(ep, score, avg, steps)

    def _adaptive_decay(self):
        # slow down decay if stuck
        if self.no_improve_counter > 30:
            self.agent.epsilon = min(0.5, self.agent.epsilon + 0.05)
            self.no_improve_counter = 0
        else:
            self.agent.decay()

    def _moving_avg(self, window):
        if len(self.scores) < window:
            return sum(self.scores) / len(self.scores)
        return sum(self.scores[-window:]) / window

    def _save_best(self, score):
        if score > self.best_score:
            self.best_score = score
            self.agent.save(MODEL_PATH)

    def _print(self, ep, score, avg, steps):
        elapsed = time.time() - self.start_time

        print(
            f"Ep {ep:4d} | "
            f"S {score:3d} | "
            f"Avg {avg:5.2f} | "
            f"Eps {self.agent.epsilon:.3f} | "
            f"Stp {steps:4d} | "
            f"Best {self.best_score:3d} | "
            f"T {elapsed:6.1f}s"
        )

    # ---------------------- Final ----------------------

    def _finalize(self):
        self.agent.save(MODEL_PATH)
        print("\nTraining Complete")
        print(f"Best Score: {self.best_score}")


# ---------------------- Entry ----------------------

def main():
    set_seed(SEED)
    ensure_dirs()

    trainer = Trainer()
    trainer.train()


if __name__ == "__main__":
    main()
