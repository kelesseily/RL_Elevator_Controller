"""
SmartLift - Q-Learning Agent
==============================
Off-policy tabular Q-learning with epsilon-greedy exploration.

Update rule:
    Q(s,a) ← Q(s,a) + α [ r + γ · max_a' Q(s',a') − Q(s,a) ]
"""

import numpy as np
import random
from collections import defaultdict


class QLearningAgent:
    """
    Tabular Q-Learning agent.

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
        n_actions: int      = 3,
        alpha: float        = 0.1,
        gamma: float        = 0.99,
        epsilon: float      = 1.0,
        epsilon_min: float  = 0.05,
        epsilon_decay: float = 0.995,
    ):
        self.n_actions     = n_actions
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table stored as defaultdict: state → numpy array of Q-values
        self.Q: dict = defaultdict(lambda: np.zeros(self.n_actions))

        # Training history
        self.episode_rewards: list[float] = []
        self.epsilons        : list[float] = []

    # ──────────────────────────────────────────────────────────────────────────
    # Policy
    # ──────────────────────────────────────────────────────────────────────────

    def select_action(self, state: tuple, greedy: bool = False) -> int:
        """
        ε-greedy action selection.
        If greedy=True, always exploit (used during evaluation).
        """
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return int(np.argmax(self.Q[state]))

    # ──────────────────────────────────────────────────────────────────────────
    # Update
    # ──────────────────────────────────────────────────────────────────────────

    def update(self, state: tuple, action: int, reward: float,
               next_state: tuple, done: bool):
        """Single Q-learning update step."""
        current_q  = self.Q[state][action]
        max_next_q = 0.0 if done else float(np.max(self.Q[next_state]))
        td_target  = reward + self.gamma * max_next_q
        td_error   = td_target - current_q
        self.Q[state][action] += self.alpha * td_error

    # ──────────────────────────────────────────────────────────────────────────
    # Epsilon Decay
    # ──────────────────────────────────────────────────────────────────────────

    def decay_epsilon(self):
        """Decay exploration rate after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.epsilons.append(self.epsilon)

    # ──────────────────────────────────────────────────────────────────────────
    # Training Loop
    # ──────────────────────────────────────────────────────────────────────────

    def train(self, env, n_episodes: int = 1000, verbose: bool = True) -> list[float]:
        """
        Run full training loop.

        Parameters
        ----------
        env        : ElevatorEnv instance
        n_episodes : number of training episodes
        verbose    : print progress every 100 episodes

        Returns
        -------
        List of total rewards per episode.
        """
        for ep in range(1, n_episodes + 1):
            state      = env.reset()
            total_reward = 0.0
            done       = False

            while not done:
                action                     = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                self.update(state, action, reward, next_state, done)
                state        = next_state
                total_reward += reward

            self.decay_epsilon()
            self.episode_rewards.append(total_reward)

            if verbose and ep % 100 == 0:
                avg = np.mean(self.episode_rewards[-100:])
                print(f"[Q-Learning] Episode {ep:>5}/{n_episodes} | "
                      f"Avg Reward (last 100): {avg:8.2f} | ε={self.epsilon:.4f} | "
                      f"States visited: {len(self.Q)}")

        return self.episode_rewards

    # ──────────────────────────────────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate(self, env, n_episodes: int = 50) -> dict:
        """Run greedy policy (no exploration) and collect metrics."""
        all_summaries = []
        for _ in range(n_episodes):
            state = env.reset()
            done  = False
            while not done:
                action = self.select_action(state, greedy=True)
                state, _, done, _ = env.step(action)
            all_summaries.append(env.get_episode_summary())

        return _aggregate_summaries(all_summaries)

    def __repr__(self):
        return (f"QLearningAgent(α={self.alpha}, γ={self.gamma}, "
                f"ε={self.epsilon:.4f}, states={len(self.Q)})")


# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────

def _aggregate_summaries(summaries: list[dict]) -> dict:
    keys = summaries[0].keys()
    return {k: float(np.mean([s[k] for s in summaries])) for k in keys}
