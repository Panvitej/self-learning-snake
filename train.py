from snake_env import SnakeEnv
from agent import Agent

MODEL_PATH = "dqn_model.pth"


def train(episodes=2000):
    env = SnakeEnv(render=False)
    agent = Agent()

    for ep in range(episodes):
        state = env.reset()
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            agent.replay()

            state = next_state

        agent.decay()

        if ep % 10 == 0:
            agent.update_target()

        print(f"Episode {ep} | Score: {env.score} | Epsilon: {agent.epsilon:.3f}")

    agent.save(MODEL_PATH)
    print("Model saved.")


if __name__ == "__main__":
    train()
