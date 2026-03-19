from snake_env import SnakeEnv
from agent import Agent

MODEL = "elite_snake.pth"


def play():
    env = SnakeEnv(render=True, speed=15)
    agent = Agent()
    agent.load(MODEL)
    agent.epsilon = 0.0

    state = env.reset()

    while True:
        action = agent.act(state)
        state, _, done = env.step(action)

        if done:
            print("Score:", env.score)
            state = env.reset()


if __name__ == "__main__":
    play()
