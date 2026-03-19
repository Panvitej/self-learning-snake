import numpy as np
from snake_env import SnakeEnv
from agent import Agent
from config import MODEL_PATH


class Evaluator:
    def __init__(self, episodes=100, render=False):
        self.episodes = episodes
        self.render = render

        self.env = SnakeEnv(render=render)
        self.agent = Agent()
        self.agent.load(MODEL_PATH)
        self.agent.epsilon = 0.0  # pure exploitation

        self.scores = []
        self.steps = []
        self.deaths = {
            "wall": 0,
            "self": 0,
            "timeout": 0
        }

    def run(self):
        for ep in range(self.episodes):
            state = self.env.reset()
            done = False
            step_count = 0

            while not done:
                action = self.agent.act(state)
                next_state, reward, done = self.env.step(action)

                state = next_state
                step_count += 1

                if done:
                    self._analyze_death(reward)

            self.scores.append(self.env.score)
            self.steps.append(step_count)

            print(f"Episode {ep:3d} | Score {self.env.score:3d} | Steps {step_count}")

        self._report()

    # -------------------- Analysis --------------------

    def _analyze_death(self, reward):
        # crude classification based on reward + position
        if reward == -10:
            head = self.env.head

            if (
                head[0] < 0 or head[0] >= self.env.WIDTH or
                head[1] < 0 or head[1] >= self.env.HEIGHT
            ):
                self.deaths["wall"] += 1
            elif head in self.env.snake[1:]:
                self.deaths["self"] += 1
            else:
                self.deaths["timeout"] += 1

    def _report(self):
        scores = np.array(self.scores)
        steps = np.array(self.steps)

        print("\n" + "=" * 50)
        print("EVALUATION SUMMARY")
        print("=" * 50)

        # --- Score metrics ---
        print("\n[Score Metrics]")
        print(f"Mean Score      : {scores.mean():.2f}")
        print(f"Median Score    : {np.median(scores):.2f}")
        print(f"Std Deviation   : {scores.std():.2f}")
        print(f"Min Score       : {scores.min()}")
        print(f"Max Score       : {scores.max()}")

        # --- Percentiles ---
        print("\n[Score Distribution]")
        for p in [25, 50, 75, 90, 95, 99]:
            print(f"P{p:<2} : {np.percentile(scores, p):.2f}")

        # --- Stability ---
        print("\n[Stability Analysis]")
        moving_avg = np.convolve(scores, np.ones(10)/10, mode='valid')
        stability = moving_avg.std()
        print(f"Moving Avg Std (window=10): {stability:.3f}")

        # --- Efficiency ---
        print("\n[Efficiency]")
        print(f"Avg Steps/Episode : {steps.mean():.2f}")
        print(f"Max Steps         : {steps.max()}")

        # --- Score per step ---
        ratio = scores / (steps + 1e-5)
        print(f"Avg Score/Step    : {ratio.mean():.4f}")

        # --- Death analysis ---
        print("\n[Failure Analysis]")
        total_deaths = sum(self.deaths.values())
        for k, v in self.deaths.items():
            perc = (v / total_deaths * 100) if total_deaths else 0
            print(f"{k.capitalize():<10}: {v} ({perc:.2f}%)")

        # --- Consistency ---
        print("\n[Consistency]")
        high_score_runs = (scores > scores.mean()).sum()
        print(f"Episodes above mean: {high_score_runs}/{len(scores)}")

        # --- Longest survival streak ---
        streak = 0
        max_streak = 0
        threshold = scores.mean()

        for s in scores:
            if s >= threshold:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0

        print(f"Longest high-performance streak: {max_streak}")

        print("\n" + "=" * 50)

    # -------------------- Optional Extensions --------------------

    def save_report(self, path="results/evaluation.txt"):
        with open(path, "w") as f:
            f.write("Evaluation Report\n")
            f.write("=================\n")
            f.write(f"Episodes: {self.episodes}\n")
            f.write(f"Mean Score: {np.mean(self.scores):.2f}\n")
            f.write(f"Max Score: {np.max(self.scores)}\n")
            f.write(f"Min Score: {np.min(self.scores)}\n")

    def get_raw_data(self):
        return {
            "scores": self.scores,
            "steps": self.steps,
            "deaths": self.deaths
        }


# -------------------- Entry --------------------

def evaluate(episodes=100, render=False):
    evaluator = Evaluator(episodes=episodes, render=render)
    evaluator.run()
    evaluator.save_report()


if __name__ == "__main__":
    evaluate()
