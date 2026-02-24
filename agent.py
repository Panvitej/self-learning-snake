import numpy as np
import random
from collections import defaultdict


class Agent:

    def __init__(self):
        self.lr = 0.1
        self.gamma = 0.9
        self.epsilon = 1.0
        self.q_table = defaultdict(lambda: np.zeros(3))

    def get_action(self, state):
        state = tuple(state)

        if random.random() < self.epsilon:
            move = random.randint(0, 2)
        else:
            move = np.argmax(self.q_table[state])

        action = [0, 0, 0]
        action[move] = 1
        return action

    def update(self, state, action, reward, next_state, done):
        state = tuple(state)
        next_state = tuple(next_state)
        action_idx = np.argmax(action)

        predict = self.q_table[state][action_idx]
        target = reward + self.gamma * np.max(self.q_table[next_state]) * (not done)

        self.q_table[state][action_idx] += self.lr * (target - predict)

    def decay_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * 0.995)
