"""
SmartLift - Visualisation Utilities
======================================
Plotting helpers for training curves, performance comparisons,
and animated elevator traces.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: smooth a reward curve
# ─────────────────────────────────────────────────────────────────────────────

def smooth(values: list[float], window: int = 50) -> np.ndarray:
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Training Curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(
    ql_rewards: list[float],
    sarsa_rewards: list[float],
    window: int = 50,
    save: bool = True,
):
    """Plot smoothed episode reward for Q-Learning vs SARSA."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    ql_smooth    = smooth(ql_rewards, window)
    sarsa_smooth = smooth(sarsa_rewards, window)
    x_ql         = np.arange(window - 1, len(ql_rewards))
    x_sarsa      = np.arange(window - 1, len(sarsa_rewards))

    ax.plot(x_ql,    ql_smooth,    color="#58a6ff", lw=2,   label="Q-Learning (smoothed)")
    ax.plot(x_sarsa, sarsa_smooth, color="#3fb950", lw=2,   label="SARSA (smoothed)")
    ax.plot(ql_rewards,    color="#58a6ff", alpha=0.15, lw=0.8)
    ax.plot(sarsa_rewards, color="#3fb950", alpha=0.15, lw=0.8)

    ax.set_xlabel("Episode",        color="white")
    ax.set_ylabel("Total Reward",   color="white")
    ax.set_title("Training Curves — Q-Learning vs SARSA", color="white", fontsize=14)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#30363d")
    ax.legend(facecolor="#161b22", labelcolor="white")
    ax.grid(color="#30363d", linewidth=0.5)

    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, "training_curves.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Performance Comparison Bar Chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_performance_comparison(metrics: dict[str, dict], save: bool = True):
    """
    Bar chart comparing key metrics across all controllers.

    Parameters
    ----------
    metrics : dict mapping controller_name → evaluation summary dict
    """
    controllers = list(metrics.keys())
    keys = [
        ("avg_waiting_time",  "Avg Waiting Time (steps)"),
        ("max_waiting_time",  "Max Waiting Time (steps)"),
        ("total_movements",   "Total Movements"),
        ("total_served",      "Passengers Served"),
    ]
    colors = ["#58a6ff", "#3fb950", "#f78166", "#d2a8ff"]

    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor("#0d1117")
    gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    for idx, (key, label) in enumerate(keys):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        ax.set_facecolor("#161b22")

        values = [metrics[c].get(key, 0) for c in controllers]
        bars   = ax.bar(controllers, values, color=colors, edgecolor="#30363d", width=0.5)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.02,
                f"{val:.1f}",
                ha="center", va="bottom", color="white", fontsize=9,
            )

        ax.set_title(label, color="white", fontsize=11)
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#30363d")
        ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1)
        ax.tick_params(axis="x", labelrotation=15)

    fig.suptitle("Controller Performance Comparison", color="white", fontsize=15, y=1.01)
    plt.tight_layout()

    if save:
        path = os.path.join(RESULTS_DIR, "performance_comparison.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Epsilon Decay Curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_epsilon_decay(ql_epsilons: list[float], sarsa_epsilons: list[float],
                       save: bool = True):
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    ax.plot(ql_epsilons,    color="#58a6ff", lw=2, label="Q-Learning ε")
    ax.plot(sarsa_epsilons, color="#3fb950", lw=2, label="SARSA ε")

    ax.set_xlabel("Episode",   color="white")
    ax.set_ylabel("Epsilon",   color="white")
    ax.set_title("Exploration Rate Decay", color="white", fontsize=13)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#30363d")
    ax.legend(facecolor="#161b22", labelcolor="white")
    ax.grid(color="#30363d", linewidth=0.5)

    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, "epsilon_decay.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Elevator Trace Animation (static snapshot)
# ─────────────────────────────────────────────────────────────────────────────

def plot_elevator_trace(trace: list[dict], n_floors: int, save: bool = True):
    """
    Plots the elevator's floor position over time alongside waiting counts.

    Parameters
    ----------
    trace    : list of info dicts recorded during one evaluation episode
    n_floors : number of floors in the building
    """
    steps          = list(range(len(trace)))
    floors         = [t["floor"]             for t in trace]
    waiting_counts = [t["waiting_total"]     for t in trace]
    inside_counts  = [t["passengers_inside"] for t in trace]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.patch.set_facecolor("#0d1117")

    for ax in (ax1, ax2):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#30363d")
        ax.grid(color="#30363d", linewidth=0.5)

    # Floor trajectory
    ax1.plot(steps, floors, color="#58a6ff", lw=1.5, label="Elevator Floor")
    ax1.set_ylabel("Floor", color="white")
    ax1.set_yticks(range(n_floors))
    ax1.set_ylim(-0.5, n_floors - 0.5)
    ax1.legend(facecolor="#161b22", labelcolor="white")
    ax1.set_title("Elevator Trace — One Evaluation Episode", color="white", fontsize=13)

    # Passenger counts
    ax2.fill_between(steps, waiting_counts, alpha=0.4, color="#f78166", label="Waiting")
    ax2.fill_between(steps, inside_counts,  alpha=0.4, color="#3fb950", label="Inside elevator")
    ax2.plot(steps, waiting_counts, color="#f78166", lw=1.2)
    ax2.plot(steps, inside_counts,  color="#3fb950", lw=1.2)
    ax2.set_xlabel("Time Step", color="white")
    ax2.set_ylabel("Passengers", color="white")
    ax2.legend(facecolor="#161b22", labelcolor="white")

    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, "elevator_trace.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Q-value Heatmap (floor vs action for fixed context)
# ─────────────────────────────────────────────────────────────────────────────

def plot_q_heatmap(agent, n_floors: int, save: bool = True):
    action_labels = ["DOWN", "STAY", "UP"]
    q_matrix = np.full((n_floors, 3), np.nan)

    # Use the most-visited state per floor from the actual Q-table
    floor_best = {}
    for state, q_vals in agent.Q.items():
        floor = state[0]
        if floor not in floor_best:
            floor_best[floor] = q_vals
        else:
            # prefer state with highest max Q-value (most learned)
            if np.max(q_vals) > np.max(floor_best[floor]):
                floor_best[floor] = q_vals

    for floor, q_vals in floor_best.items():
        q_matrix[floor] = q_vals

    # Fill any unvisited floors with zeros
    q_matrix = np.nan_to_num(q_matrix, nan=0.0)

    fig, ax = plt.subplots(figsize=(6, 8))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    vabs = max(abs(q_matrix.min()), abs(q_matrix.max())) or 1
    im = ax.imshow(q_matrix, aspect="auto", cmap="RdYlGn",
                   vmin=-vabs, vmax=vabs)

    ax.set_xticks(range(3))
    ax.set_xticklabels(action_labels, color="white")
    ax.set_yticks(range(n_floors))
    ax.set_yticklabels([f"Floor {i}" for i in range(n_floors)], color="white")
    ax.set_title(f"Q-Value Heatmap — {agent.__class__.__name__}\n(best learned state per floor)",
                 color="white", fontsize=12)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#30363d")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03)
    cbar.ax.tick_params(colors="white")
    cbar.ax.yaxis.label.set_color("white")

    plt.tight_layout()
    if save:
        name = agent.__class__.__name__.lower()
        path = os.path.join(RESULTS_DIR, f"q_heatmap_{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.show()
