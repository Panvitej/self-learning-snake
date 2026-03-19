# Self-Learning Snake: Deep Reinforcement Learning System

## 1. Problem Definition

The objective is to train an autonomous agent to play Snake optimally.

Constraints:
- No prior knowledge of the environment
- No hardcoded rules or heuristics
- Learning must emerge purely from interaction

Core challenge:
- Sequential decision-making under uncertainty
- Sparse rewards (food vs death)
- Long-term survival vs short-term gain

---

## 2. System Overview

The system consists of two tightly coupled components:

### Environment (snake_env.py)
- Deterministic grid-based simulation
- Provides state, reward, and termination signal
- Enforces physics and constraints

### Agent (agent.py)
- Learns a policy π(s) → a
- Approximates Q-values using a neural network
- Improves through replayed experience

### Training Loop (train.py)
- Drives interaction between agent and environment
- Handles learning updates and exploration schedule

### Inference (play.py)
- Executes trained policy with ε = 0
- No learning, only exploitation

---

## 3. State Representation

The environment is reduced to a **7-dimensional binary feature vector**:

| Feature | Meaning |
|--------|--------|
| Danger straight | Collision risk ahead |
| Danger right | Collision risk if turning right |
| Danger left | Collision risk if turning left |
| Food left | Food is left of head |
| Food right | Food is right of head |
| Food up | Food is above |
| Food down | Food is below |

Design rationale:
- Minimal representation → faster convergence
- Removes irrelevant spatial complexity
- Encodes only decision-critical information

Tradeoff:
- No full spatial awareness (limits optimality ceiling)

---

## 4. Action Space

Discrete action space:

| Action | Description |
|-------|------------|
| 0 | Move straight |
| 1 | Turn right |
| 2 | Turn left |

Important:
- Actions are **relative**, not absolute
- Reduces action ambiguity and state complexity

---

## 5. Reward Design

| Event | Reward |
|------|--------|
| Eat food | +10 |
| Collision | -10 |
| Each step | -0.1 |

Design logic:
- Positive reinforcement for goal completion
- Strong penalty for failure
- Small negative step cost prevents idle looping

Impact:
- Encourages shortest-path behavior
- Reduces wandering
- Improves convergence speed

---

## 6. Learning Architecture

The agent uses an advanced variant of Deep Q-Learning:

### 6.1 Q-Function Approximation

Instead of a table:
- Q(s, a) is approximated using a neural network

Input: state (7 features)  
Output: Q-values for 3 actions  

---

### 6.2 Double DQN

Problem addressed:
- Standard DQN overestimates Q-values

Solution:
- Separate networks for:
  - Action selection (online network)
  - Action evaluation (target network)

Effect:
- More stable and realistic value estimates

---

### 6.3 Dueling Network Architecture

Network splits into:

- Value stream: V(s)
- Advantage stream: A(s, a)

Final Q-value:
Q(s, a) = V(s) + A(s, a) − mean(A)

Benefit:
- Learns state importance independent of action
- Faster policy stabilization

---

### 6.4 Prioritized Experience Replay (PER)

Standard replay:
- Uniform random sampling

PER:
- Samples based on TD-error magnitude

Effect:
- Important transitions are replayed more often
- Faster correction of mistakes

---

### 6.5 N-Step Learning (n = 3)

Instead of single-step reward:
- Accumulates reward over multiple steps

R = r₀ + γr₁ + γ²r₂

Benefit:
- Better long-term credit assignment
- Improves learning in delayed reward scenarios

---

## 7. Training Process

Each episode:

1. Environment reset
2. Agent observes initial state
3. Loop until termination:
   - Select action (ε-greedy)
   - Execute action
   - Receive reward and next state
   - Store transition
   - Perform training step
4. Decay exploration rate

---

## 8. Exploration Strategy

ε-greedy policy:

- Initially:
  - ε = 1.0 → full exploration
- Gradually:
  - ε decays toward 0.02
- Finally:
  - Mostly exploitation

Purpose:
- Avoid premature convergence
- Ensure sufficient state coverage

---

## 9. Stability Mechanisms

To prevent divergence:

- Target network (soft updates)
- Gradient clipping
- Replay buffer (decorrelated data)
- Controlled learning rate

These are critical for deep RL stability.

---

## 10. Model Persistence

- Model saved using PyTorch
- File: `snake_model.pth`

Enables:
- Reuse of trained policy
- Deployment without retraining

---

## 11. Performance Evolution

### Early Training
- Random movement
- High collision rate
- Low scores

### Mid Training
- Basic obstacle avoidance
- Occasional food targeting

### Late Training
- Structured navigation
- Efficient food acquisition
- Consistent scoring patterns

---

## 12. Design Tradeoffs

| Decision | Benefit | Limitation |
|--------|--------|-----------|
| Compact state | Fast learning | Limited awareness |
| DQN | Generalization | Requires tuning |
| PER | Faster convergence | Added complexity |
| N-step | Better rewards | More computation |

---

## 13. System Limitations

- No full-grid representation
- Cannot plan long trajectories explicitly
- Performance bounded by state abstraction

---

## 14. Extensions (Next-Level Systems)

To push beyond this system:

- CNN-based input (full grid vision)
- Parallel environments (multi-agent training)
- Distributed RL training
- Policy gradient methods (PPO, A3C)

---

## 15. Summary

This system demonstrates:

- End-to-end reinforcement learning pipeline
- Transition from tabular → deep RL
- Practical implementation of:
  - Double DQN
  - Dueling networks
  - Prioritized replay
  - N-step returns

Outcome:
A self-learning agent that develops structured, goal-oriented behavior purely from interaction.

No hardcoded strategy. Only learning.
