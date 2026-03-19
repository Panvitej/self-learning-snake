import numpy as np
import time

from snake_env import SnakeEnv
from agent import Agent
from config import MODEL_PATH


class Player:
    def __init__(self, speed=15):
        self.env = SnakeEnv(render=True, speed=speed)
        self.agent = Agent()
        self.agent.load(MODEL_PATH)

        self.agent.epsilon = 0.0

        self.scores = []
        self.games = 0
        self.start_time = time.time()

    # ---------------------- Main Loop ----------------------

    def run(self):
        state = self.env.reset()

        while True:
            action = self.agent.act(state)

            # no unnecessary unpacking overhead
            result = self.env.step(action)
            state = result[0]
            done = result[2]

            if done:
                self._end_game()
                state = self.env.reset()

    # ---------------------- Optimized Tracking ----------------------

    def _end_game(self):
        score = self.env.score
        self.scores.append(score)
        self.games += 1

        if self.games % 5 == 0:
            self._summary()

        print(f"Game {self.games:3d} | Score {score:3d}")

    def _summary(self):
        scores = np.array(self.scores)

        avg = scores.mean()
        best = scores.max()

        # trend (last 10 games)
        recent = scores[-10:] if len(scores) >= 10 else scores
        trend = recent.mean()

        print(
            f"[Summary] "
            f"Avg {avg:.2f} | "
            f"Best {best} | "
            f"Recent {trend:.2f}"
        )


# ---------------------- Entry ----------------------

def main():
    player = Player(speed=15)
    player.run()


if __name__ == "__main__":
    main()
