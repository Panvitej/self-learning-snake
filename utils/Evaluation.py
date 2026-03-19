from snake_env import SnakeEnv
from agent import Agent
from config import MODEL_PATH


def evaluate(episodes=100):
    env = SnakeEnv(render=False)
    agent = Agent()
    agent.load(MODEL_PATH)
    agent.epsilon = 0.0

    scores = []

    for _ in range(episodes):
        state = env.reset()
        done = False

        while not done:
            action = agent.act(state)
            state, _, done = env.step(action)

        scores.append(env.score)

    avg_score = sum(scores) / len(scores)
    max_score = max(scores)
    min_score = min(scores)

    print("Evaluation Results")
    print(f"Average Score: {avg_score:.2f}")
    print(f"Max Score: {max_score}")
    print(f"Min Score: {min_score}")


if __name__ == "__main__":
    evaluate()
