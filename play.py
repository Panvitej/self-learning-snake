import pickle
from snake_env import SnakeEnv
from agent import Agent


MODEL_PATH = "q_table.pkl"


def load_agent():
    agent = Agent()
    agent.epsilon = 0.0  # pure exploitation

    try:
        with open(MODEL_PATH, "rb") as f:
            agent.q_table = pickle.load(f)
        print("Model loaded.")
    except FileNotFoundError:
        print("No trained model found. Run train.py first.")
        exit()

    return agent


def main():
    env = SnakeEnv()
    agent = load_agent()

    state = env.reset()
    done = False

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        state = next_state

        if done:
            print("Game Over. Score:", env.score)
            state = env.reset()


if __name__ == "__main__":
    main()
