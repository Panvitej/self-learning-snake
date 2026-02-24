from snake_env import SnakeEnv
from agent import Agent
import pickle

env = SnakeEnv()
agent = Agent()

episodes = 1000

for episode in range(episodes):
    state = env.reset()
    done = False

    while not done:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state

    agent.decay_epsilon()

    print(f"Episode {episode}, Score: {env.score}, Epsilon: {agent.epsilon:.3f}")
    
with open("q_table.pkl", "wb") as f:
    pickle.dump(agent.q_table, f)

print("Model saved.")
