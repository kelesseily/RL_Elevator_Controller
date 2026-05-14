"""
SmartLift — Main Training & Evaluation Script
===============================================
Run this file to:
  1. Train Q-Learning and SARSA agents on the elevator environment
  2. Evaluate all controllers (Q-Learning, SARSA, Nearest-Request, FCFS)
  3. Generate and save all plots to results/

Usage
-----
    python main.py                  # default settings
    python main.py --episodes 2000  # custom episode count
    python main.py --floors 6       # smaller building (faster)
    python main.py --no-plots       # skip matplotlib (headless)
"""

import argparse
import sys
import os
import numpy as np

# Make sure local packages are importable when running as a script
sys.path.insert(0, os.path.dirname(__file__))

from environment.elevator_env import ElevatorEnv
from agents.q_learning          import QLearningAgent
from agents.sarsa               import SARSAAgent
from baselines.rule_based       import NearestRequestController, FCFSController, MPCController
from utils.visualise            import (
    plot_training_curves,
    plot_performance_comparison,
    plot_epsilon_decay,
    plot_elevator_trace,
    plot_q_heatmap,
)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SmartLift RL Elevator Controller")
    p.add_argument("--floors",       type=int,   default=10,    help="Number of building floors")
    p.add_argument("--capacity",     type=int,   default=8,     help="Max elevator capacity")
    p.add_argument("--arrival-rate", type=float, default=0.3,   help="Poisson arrival rate")
    p.add_argument("--max-steps",    type=int,   default=500,   help="Steps per episode")
    p.add_argument("--episodes",     type=int,   default=5000,  help="Training episodes")
    p.add_argument("--eval-episodes",type=int,   default=50,    help="Evaluation episodes")
    p.add_argument("--alpha",        type=float, default=0.1,   help="Learning rate")
    p.add_argument("--gamma",        type=float, default=0.99,  help="Discount factor")
    p.add_argument("--epsilon",      type=float, default=1.0,   help="Initial epsilon")
    p.add_argument("--epsilon-min",  type=float, default=0.05,  help="Minimum epsilon")
    p.add_argument("--epsilon-decay",type=float, default=0.995, help="Epsilon decay")
    p.add_argument("--seed",         type=int,   default=42,    help="Random seed")
    p.add_argument("--no-plots",     action="store_true",       help="Disable plot generation")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_env(args, seed_offset: int = 0) -> ElevatorEnv:
    return ElevatorEnv(
        n_floors      = args.floors,
        max_capacity  = args.capacity,
        arrival_rate  = args.arrival_rate,
        max_steps     = args.max_steps,
        seed          = args.seed + seed_offset,
    )


def record_trace(agent_or_controller, env: ElevatorEnv) -> list[dict]:
    """Run one episode and record per-step info for trace plot."""
    trace = []
    from agents.q_learning import QLearningAgent
    from agents.sarsa import SARSAAgent

    if isinstance(agent_or_controller, (QLearningAgent, SARSAAgent)):
        # RL agent expects state tuple
        state = env.reset()
        done  = False
        while not done:
            action = agent_or_controller.select_action(state, greedy=True)
            state, _, done, info = env.step(action)
            trace.append(info)
    else:
        # Baseline controller expects env object
        env.reset()
        done = False
        while not done:
            action = agent_or_controller.select_action(env)
            _, _, done, info = env.step(action)
            trace.append(info)
    return trace


def print_metrics_table(metrics: dict[str, dict]):
    header = f"{'Controller':<22} {'Served':>8} {'Avg Wait':>10} {'Max Wait':>10} {'Movements':>11} {'Avg Total':>11}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for name, m in metrics.items():
        print(
            f"{name:<22} "
            f"{m['total_served']:>8.1f} "
            f"{m['avg_waiting_time']:>10.2f} "
            f"{m['max_waiting_time']:>10.2f} "
            f"{m['total_movements']:>11.1f} "
            f"{m['avg_total_time']:>11.2f}"
        )
    print("=" * len(header) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 60)
    print("  SmartLift — RL-Based Elevator Control System")
    print("=" * 60)
    print(f"  Floors      : {args.floors}")
    print(f"  Capacity    : {args.capacity}")
    print(f"  Arrival rate: {args.arrival_rate}")
    print(f"  Steps/ep    : {args.max_steps}")
    print(f"  Episodes    : {args.episodes}")
    print("=" * 60)

    # ── Environments ─────────────────────────────────────────────────────────
    train_env_ql    = make_env(args, seed_offset=0)
    train_env_sarsa = make_env(args, seed_offset=1)
    eval_env        = make_env(args, seed_offset=99)

    # ── Shared hyperparameters ────────────────────────────────────────────────
    hp = dict(
        n_actions     = 3,
        alpha         = args.alpha,
        gamma         = args.gamma,
        epsilon       = args.epsilon,
        epsilon_min   = args.epsilon_min,
        epsilon_decay = args.epsilon_decay,
    )

    # ── Train Q-Learning ──────────────────────────────────────────────────────
    print("\n[1/2] Training Q-Learning Agent …")
    ql_agent = QLearningAgent(**hp)
    ql_rewards = ql_agent.train(train_env_ql, n_episodes=args.episodes)
    print(f"  → Q-table size: {len(ql_agent.Q)} states")

    # ── Train SARSA ───────────────────────────────────────────────────────────
    print("\n[2/2] Training SARSA Agent …")
    sarsa_agent = SARSAAgent(**hp)
    sarsa_rewards = sarsa_agent.train(train_env_sarsa, n_episodes=args.episodes)
    print(f"  → Q-table size: {len(sarsa_agent.Q)} states")

    # ── Evaluate all controllers ──────────────────────────────────────────────
    print("\nEvaluating all controllers …")

    nearest_ctrl = NearestRequestController()
    fcfs_ctrl    = FCFSController()
    mpc_ctrl     = MPCController(horizon=6)

    metrics = {}
    metrics["Q-Learning"]      = ql_agent.evaluate(eval_env, n_episodes=args.eval_episodes)
    metrics["SARSA"]           = sarsa_agent.evaluate(eval_env, n_episodes=args.eval_episodes)
    metrics["Nearest-Request"] = nearest_ctrl.evaluate(eval_env, n_episodes=args.eval_episodes)
    metrics["FCFS"]            = fcfs_ctrl.evaluate(eval_env, n_episodes=args.eval_episodes)
    metrics["MPC"]             = mpc_ctrl.evaluate(eval_env, n_episodes=args.eval_episodes)
    print_metrics_table(metrics)

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not args.no_plots:
        print("Generating plots …")

        plot_training_curves(ql_rewards, sarsa_rewards)
        plot_performance_comparison(metrics)
        plot_epsilon_decay(ql_agent.epsilons, sarsa_agent.epsilons)

        # Trace for Q-Learning agent
        trace_env = make_env(args, seed_offset=999)
        ql_trace  = record_trace(ql_agent, trace_env)
        plot_elevator_trace(ql_trace, n_floors=args.floors)

        # Q-value heatmaps
        plot_q_heatmap(ql_agent,    n_floors=args.floors)
        plot_q_heatmap(sarsa_agent, n_floors=args.floors)

        print(f"\nAll plots saved to: {os.path.abspath('results/')}")

    print("\nDone. ✓")


if __name__ == "__main__":
    main()
