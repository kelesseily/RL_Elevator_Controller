"""
SmartLift - SARSA Agent
=========================
On-policy tabular SARSA with epsilon-greedy exploration.

Update rule:
    Q(s,a) ← Q(s,a) + α [ r + γ · Q(s',a') − Q(s,a) ]

Key difference from Q-learning: uses the *actual* next action a' (chosen by
the current policy) rather than the *greedy* max action.
"""

import numpy as np
import random
from collections import defaultdict


class SARSAAgent:
    """
    Tabular SARSA agent.

    Parameters
    ----------
    n_actions       : size of the discrete action space
    alpha           : learning rate  (α)
    gamma           : discount factor (γ)
    epsilon         : initial exploration rate
    epsilon_min     : minimum exploration rate
    epsilon_decay   : multiplicative decay applied after each episode
    """

    def __init__(
        self,
        n_actions: int       = 3,
        alpha: float         = 0.1,
        gamma: float         = 0.99,
        epsilon: float       = 1.0,
        epsilon_min: float   = 0.05,
        epsilon_decay: float = 0.9995,
        current_episode: int  = 0,
    ):
        self.n_actions     = n_actions
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.current_episode = current_episode

        self.Q: dict = defaultdict(lambda: np.zeros(self.n_actions))

        self.episode_rewards: list[float] = []
        self.epsilons        : list[float] = []

    # ──────────────────────────────────────────────────────────────────────────
    # Policy
    # ──────────────────────────────────────────────────────────────────────────

    def select_action(self, state: tuple, greedy: bool = False) -> int:
        """ε-greedy action selection."""
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return int(np.argmax(self.Q[state]))

    # ──────────────────────────────────────────────────────────────────────────
    # Update
    # ──────────────────────────────────────────────────────────────────────────

    def update(self, state: tuple, action: int, reward: float,
               next_state: tuple, next_action: int, done: bool):
        """
        SARSA update — uses the on-policy next action (not max).
        """
        alpha = max(0.01, self.alpha * (0.9995 ** self.current_episode))
        current_q  = self.Q[state][action]
        next_q     = 0.0 if done else self.Q[next_state][next_action]
        td_target  = reward + self.gamma * next_q
        td_error   = td_target - current_q
        self.Q[state][action] += alpha * td_error

    # ──────────────────────────────────────────────────────────────────────────
    # Epsilon Decay
    # ──────────────────────────────────────────────────────────────────────────

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.epsilons.append(self.epsilon)
        self.current_episode += 1

    # ──────────────────────────────────────────────────────────────────────────
    # Training Loop
    # ──────────────────────────────────────────────────────────────────────────

    def train(self, env, n_episodes: int = 1000, verbose: bool = True) -> list[float]:
        """
        Run full SARSA training loop.

        SARSA requires choosing a' BEFORE updating, so the inner loop
        structure differs slightly from Q-learning.
        """
        for ep in range(1, n_episodes + 1):
            state        = env.reset()
            action       = self.select_action(state)
            total_reward = 0.0
            done         = False

            while not done:
                next_state, reward, done, _ = env.step(action)
                next_action = self.select_action(next_state)

                self.update(state, action, reward, next_state, next_action, done)

                state        = next_state
                action       = next_action
                total_reward += reward

            self.decay_epsilon()
            self.episode_rewards.append(total_reward)

            if verbose and ep % 100 == 0:
                avg = np.mean(self.episode_rewards[-100:])
                print(f"[SARSA]      Episode {ep:>5}/{n_episodes} | "
                      f"Avg Reward (last 100): {avg:8.2f} | ε={self.epsilon:.4f} | "
                      f"States visited: {len(self.Q)}")

        return self.episode_rewards

    # ──────────────────────────────────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate(self, env, n_episodes: int = 50) -> dict:
        """Run greedy policy and collect metrics."""
        all_summaries = []
        for _ in range(n_episodes):
            state = env.reset()
            done  = False
            while not done:
                action = self.select_action(state, greedy=True)
                state, _, done, _ = env.step(action)
            all_summaries.append(env.get_episode_summary())

        keys = all_summaries[0].keys()
        return {k: float(np.mean([s[k] for s in all_summaries])) for k in keys}

    def __repr__(self):
        return (f"SARSAAgent(α={self.alpha}, γ={self.gamma}, "
                f"ε={self.epsilon:.4f}, states={len(self.Q)})")
