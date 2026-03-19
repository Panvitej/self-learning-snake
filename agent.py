import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


# ---------- Prioritized Replay ----------
class PERBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0

    def add(self, transition):
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        prios = self.priorities[:len(self.buffer)]
        probs = prios ** self.alpha
        probs /= probs.sum()

        idxs = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in idxs]

        weights = (len(self.buffer) * probs[idxs]) ** (-beta)
        weights /= weights.max()

        return samples, idxs, np.array(weights, dtype=np.float32)

    def update_priorities(self, idxs, errors):
        for i, err in zip(idxs, errors):
            self.priorities[i] = abs(err) + 1e-5


# ---------- Dueling Network ----------
class Network(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.adv_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)
        adv = self.adv_stream(x)
        return value + adv - adv.mean(dim=1, keepdim=True)


# ---------- Agent ----------
class Agent:
    def __init__(self, state_size=7, action_size=3):
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = 0.9
        self.lr = 0.0005

        self.epsilon = 1.0
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.995

        self.batch_size = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = Network(state_size, action_size).to(self.device)
        self.target = Network(state_size, action_size).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.memory = PERBuffer(100_000)

        # N-step
        self.n_step = 3
        self.n_buffer = deque(maxlen=self.n_step)

        self._sync_target()

    def _sync_target(self):
        self.target.load_state_dict(self.model.state_dict())

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return int(torch.argmax(self.model(state)).item())

    # ---------- N-step logic ----------
    def remember(self, s, a, r, ns, done):
        self.n_buffer.append((s, a, r, ns, done))

        if len(self.n_buffer) < self.n_step:
            return

        R = sum([self.n_buffer[i][2] * (self.gamma ** i) for i in range(self.n_step)])
        s0, a0 = self.n_buffer[0][:2]
        ns_last, done_last = self.n_buffer[-1][3], self.n_buffer[-1][4]

        self.memory.add((s0, a0, R, ns_last, done_last))

    # ---------- Training ----------
    def train_step(self, beta=0.4):
        if len(self.memory.buffer) < self.batch_size:
            return

        batch, idxs, weights = self.memory.sample(self.batch_size, beta)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        weights = torch.tensor(weights).to(self.device)

        q = self.model(states)
        q = q.gather(1, actions.unsqueeze(1)).squeeze(1)

        next_actions = torch.argmax(self.model(next_states), dim=1)
        next_q = self.target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)

        target = rewards + self.gamma * next_q * (1 - dones)

        loss = (weights * (q - target.detach()) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        errors = (q - target).detach().cpu().numpy()
        self.memory.update_priorities(idxs, errors)

        self._soft_update()

    def _soft_update(self, tau=0.01):
        for t, s in zip(self.target.parameters(), self.model.parameters()):
            t.data.copy_(tau * s.data + (1 - tau) * t.data)

    def decay(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        self._sync_target()
