# Self-Learning Snake (Q-Learning)

A reinforcement learning implementation of the classic Snake game using tabular Q-learning.  
The agent learns autonomously through reward-driven interaction with the environment, improving from random exploration to goal-directed behavior.

---

## Overview

This project includes:

- Custom Snake game environment (Pygame)
- Discrete state representation
- Tabular Q-learning agent
- Exploration-to-exploitation training schedule
- Model saving and playback mode

The snake begins with no prior knowledge. Through repeated episodes, it learns to avoid collisions and maximize reward by collecting food.

---

## State Representation

The agent observes:

- Danger straight  
- Danger right  
- Danger left  
- Food location (left/right/up/down relative to head)  

This compact state encoding reduces the environment to a manageable discrete space.

---

## Action Space

The agent chooses one of three actions:

- Move straight  
- Turn right  
- Turn left  

Actions are relative to the current direction.

---

## Reward Function

- +10 → Food collected  
- -10 → Collision (game over)  
- 0 → Normal movement  

This reward structure encourages survival and goal-seeking behavior.

---

## Learning Algorithm

Q-learning update rule:

Q(s, a) ← Q(s, a) + α [ r + γ max Q(s', a') − Q(s, a) ]

Where:

- α = learning rate  
- γ = discount factor  
- ε = exploration rate  

Exploration starts high and decays gradually, shifting toward exploitation of the learned policy.

---

## Project Structure
snake-rl/
│
├── agent.py
├── snake_env.py
├── train.py
├── play.py
├── models/
│ └── q_table.pkl
└── requirements.txt

## Installation


pip install -r requirements.txt

Dependencies:

- pygame
- numpy

---

## Training

Run:


python train.py

During training:

- The agent explores randomly at first.
- Q-values are updated after each action.
- Exploration rate decays gradually.
- Performance improves across episodes.

At completion, the learned Q-table is saved to:


models/q_table.pkl


---

## Play (Exploitation Mode)

After training:


python play.py


The agent runs in pure exploitation mode (ε = 0) and follows the learned policy.

---

## Learning Behavior

Early episodes:
- Frequent collisions
- Random movement
- Low scores

Mid training:
- Improved wall avoidance
- More consistent food targeting

Later episodes:
- Structured movement patterns
- Higher average survival time
- More stable policy execution

---

## Design Decisions

- Tabular Q-learning for interpretability
- Discrete state space for stability
- Modular separation between environment and agent
- Model persistence using pickle

---

## Limitations

- Tabular approach does not scale to large state spaces
- No neural network generalization (not a DQN)
- Limited state encoding (no full board awareness)

---

## Future Improvements

- Deep Q-Network (DQN)
- Experience replay
- Headless training mode
- Performance visualization
- Hyperparameter tuning experiments

---

## Summary

This project demonstrates:

- Reinforcement learning fundamentals
- Environment-agent abstraction
- Reward-driven optimization
- Exploration vs exploitation tradeoff
- Model persistence and evaluation
