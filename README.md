# SmartLift — RL Elevator Controller

> Tabular Reinforcement Learning applied to multi-floor elevator dispatch.  
> Trains Q-Learning and SARSA agents, benchmarks them against MPC, FCFS, and Nearest-Request baselines, and ships a fully self-contained browser visualization.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Environment (MDP)](#environment-mdp)
4. [Controllers](#controllers)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Outputs](#outputs)
8. [Visualization](#visualization)
9. [Results](#results)
10. [Design Decisions](#design-decisions)

---

## Overview

SmartLift models elevator dispatch as a Markov Decision Process and trains two tabular RL agents — Q-Learning (off-policy) and SARSA (on-policy) — to minimize passenger waiting time while limiting unnecessary movement. Three classical baselines (Nearest-Request, FCFS, MPC) are included for comparison.

The project is fully self-contained: no deep learning frameworks, no gym dependency. Everything runs on `numpy` + `matplotlib`.

---

## Project Structure

```
RL_Elevator_Controller/
├── main.py                        # Training, evaluation, and plot generation
├── requirements.txt               # numpy, matplotlib
│
├── environment/
│   └── elevator_env.py            # MDP: ElevatorEnv + Passenger model
│
├── agents/
│   ├── q_learning.py              # Off-policy Q-Learning agent
│   └── sarsa.py                   # On-policy SARSA agent
│
├── baselines/
│   └── rule_based.py              # NearestRequest, FCFS, MPC controllers
│
├── utils/
│   └── visualise.py               # Matplotlib plot helpers
│
├── results/                       # Auto-generated outputs
│   ├── training_curves.png
│   ├── performance_comparison.png
│   ├── epsilon_decay.png
│   ├── elevator_trace.png
│   ├── q_heatmap_qlearningagent.png
│   ├── q_heatmap_sarsaagent.png
│   ├── qtable_qlearning.json      # Exported Q-table (loadable into viz)
│   └── qtable_sarsa.json
│
└── smartlift_viz.html             # Browser visualization (standalone)
```

---

## Environment (MDP)

**File:** `environment/elevator_env.py`

### State Space

Each state is an 8-tuple with binary/bucketed signals — kept small deliberately so tabular RL remains tractable:

| Index | Field | Values | Meaning |
|-------|-------|--------|---------|
| 0 | `floor` | 0–9 | Current elevator floor |
| 1 | `direction` | 0, 1, 2 | DOWN / IDLE / UP |
| 2 | `calls_above` | 0, 1 | Any waiting passenger above current floor |
| 3 | `calls_below` | 0, 1 | Any waiting passenger below current floor |
| 4 | `calls_here` | 0, 1 | Waiting passenger at current floor |
| 5 | `dest_above` | 0, 1 | Passenger inside with destination above |
| 6 | `dest_below` | 0, 1 | Passenger inside with destination below |
| 7 | `load` | 0, 1, 2 | Empty / Partial / Full |

Maximum theoretical states: 10 × 3 × 2⁶ = **1,920**. In practice ~800–900 states are visited during training.

### Actions

| Value | Name | Effect |
|-------|------|--------|
| 0 | DOWN | Move one floor down (no-op at floor 0) |
| 1 | STAY | Serve current floor (board/alight passengers) |
| 2 | UP | Move one floor up (no-op at top floor) |

### Reward Function

```
reward = +10 × passengers_delivered
       -  0.5 × (1 if moved else 0)
       -  1.0 × min(total_waiting_passengers, 10)
```

The waiting penalty is capped at 10 to prevent reward divergence when queues are long.

### Passenger Arrivals

Passengers arrive each step according to a Poisson process with rate `λ = 0.3`. Each passenger has a random origin and a distinct random destination.

### Default Parameters

| Parameter | Default | CLI flag |
|-----------|---------|----------|
| Floors | 10 | `--floors` |
| Capacity | 8 | `--capacity` |
| Arrival rate | 0.3 | `--arrival-rate` |
| Steps per episode | 500 | `--max-steps` |
| Training episodes | 5000 | `--episodes` |
| Evaluation episodes | 50 | `--eval-episodes` |

---

## Controllers

### Q-Learning (`agents/q_learning.py`)

Off-policy tabular Q-Learning. The update rule uses the greedy maximum over next-state Q-values regardless of the actual action taken:

```
Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') − Q(s,a)]
```

The learning rate decays per episode as `α_t = max(0.01, α₀ × 0.9995^t)` to stabilize late training.

### SARSA (`agents/sarsa.py`)

On-policy SARSA. Uses the *actual* next action selected by the current ε-greedy policy, making it more conservative than Q-Learning:

```
Q(s,a) ← Q(s,a) + α · [r + γ · Q(s',a') − Q(s,a)]
```

Both agents share the same ε-greedy exploration schedule (ε decays from 1.0 → 0.05).

### Nearest-Request (`baselines/rule_based.py`)

Greedy heuristic. Priority order:
1. Serve current floor if anyone needs to board or alight.
2. Deliver passengers inside to their nearest destination.
3. Move toward the nearest waiting floor.

### FCFS — First-Come-First-Served (`baselines/rule_based.py`)

Tracks the oldest unanswered hall call by arrival timestamp and drives directly to it before attending newer calls. Car calls (destinations of passengers already inside) take priority over new hall calls.

### MPC — Model Predictive Control (`baselines/rule_based.py`)

At each step, simulates all three possible actions over a 6-step lookahead horizon using a lightweight internal model (no new arrivals assumed during rollout). Selects the first action of the sequence with the highest discounted reward. No training required — plans entirely online.

---

## Installation

```bash
# Clone / unzip the project
cd RL_Elevator_Controller

# Install dependencies (Python 3.10+)
pip install -r requirements.txt
```

No other dependencies. The visualization is a standalone HTML file with no server needed.

---

## Usage

### Train and evaluate (all defaults)

```bash
python main.py
```

### Common options

```bash
# Fewer episodes for a quick test
python main.py --episodes 500 --eval-episodes 10

# Smaller building (faster training)
python main.py --floors 6

# Skip matplotlib (headless / CI)
python main.py --no-plots

# Custom hyperparameters
python main.py --alpha 0.05 --gamma 0.95 --epsilon-decay 0.999
```

### Full CLI reference

```
--floors          Number of building floors            (default: 10)
--capacity        Max elevator capacity                (default: 8)
--arrival-rate    Poisson arrival rate per step        (default: 0.3)
--max-steps       Steps per episode                    (default: 500)
--episodes        Training episodes                    (default: 5000)
--eval-episodes   Evaluation episodes                  (default: 50)
--alpha           Learning rate α                      (default: 0.1)
--gamma           Discount factor γ                    (default: 0.99)
--epsilon         Initial exploration rate ε           (default: 1.0)
--epsilon-min     Minimum ε                            (default: 0.05)
--epsilon-decay   Multiplicative ε decay per episode   (default: 0.995)
--seed            Random seed                          (default: 42)
--no-plots        Disable matplotlib output
```

---

## Outputs

After `python main.py` completes, the `results/` folder contains:

| File | Description |
|------|-------------|
| `training_curves.png` | Smoothed episode reward — Q-Learning vs SARSA |
| `performance_comparison.png` | Bar chart comparing all 5 controllers across key metrics |
| `epsilon_decay.png` | ε schedule over training episodes |
| `elevator_trace.png` | Floor-by-floor trace of a Q-Learning evaluation episode |
| `q_heatmap_qlearningagent.png` | Q-value heatmap across states |
| `q_heatmap_sarsaagent.png` | Q-value heatmap for SARSA |
| `qtable_qlearning.json` | Serialized Q-table (string keys, list values) |
| `qtable_sarsa.json` | Serialized SARSA Q-table |

The JSON Q-tables can be loaded directly into the browser visualization without re-running training.

---

## Visualization

`smartlift_viz.html` is a standalone browser app — open it directly, no server required.

### Features

- **Live elevator animation** — the cab moves smoothly up and down the shaft; doors open when boarding/alighting
- **Waiting passengers** — animated dots appear on each floor with a queue count
- **5 controller tabs** — switch between Q-Learning, SARSA, MPC, FCFS, and Nearest-Request mid-session
- **Decision panel** — shows the three Q-values (↓ STAY ↑) and the state encoding at each step; highlights the chosen action
- **Episode metrics** — live moves, average wait time, and a served/moves efficiency ratio
- **Comparison chart** — accumulates served-passenger counts across all controllers you run, rendered as a bar chart after each episode
- **Speed control** — 1×, 2×, 4×, 8×
- **Custom Q-table loader** — drag in your own `qtable_qlearning.json` or `qtable_sarsa.json` to visualize a freshly trained model

### State display legend

```
F5 dir:↑ calls↑1↓0h0 ins↑1↓0 load:½
│    │         │            │       └─ elevator load (∅ / ½ / F)
│    │         │            └───────── destinations inside (above / below)
│    │         └────────────────────── hall calls (above / below / here)
│    └──────────────────────────────── direction
└───────────────────────────────────── current floor
```

`[fallback]` appears when the current state was never visited during training — the visualizer falls back to the nearest-request heuristic automatically.

---

## Results

Typical evaluation results after 5,000 training episodes (50-episode average, 10-floor building):

| Controller | Avg Served | Avg Wait | Avg Movements |
|------------|-----------|----------|---------------|
| Q-Learning | ~85 | ~4.2s | ~310 |
| SARSA | ~83 | ~4.5s | ~305 |
| MPC (H=6) | ~80 | ~5.1s | ~290 |
| Nearest-Request | ~76 | ~5.8s | ~340 |
| FCFS | ~74 | ~6.3s | ~350 |

RL agents outperform all rule-based baselines on passengers served and average wait time. SARSA is slightly more movement-efficient due to its conservative on-policy updates; Q-Learning tends to serve marginally more passengers.

---

## Design Decisions

**Why tabular RL?** The state space (< 2,000 states) is small enough that a lookup table converges reliably and remains fully interpretable — you can inspect every Q-value directly.

**Why binary state signals instead of bitmasks?** Full bitmasks over 10 floors give 2¹⁰ = 1,024 combinations per call direction, making the table 50× larger without meaningfully improving policy quality for a single elevator. Binary "calls above / below / here" captures the actionable information.

**Why cap the waiting penalty at 10?** Without a cap, a large sudden queue spike can produce reward values orders of magnitude larger than normal, destabilizing early training before the agent has built up a reasonable policy.

**Why decay α per episode?** A fixed learning rate can oscillate in late training when the policy is near-optimal. The decaying schedule `α_t = max(0.01, α₀ × 0.9995^t)` lets the agent learn quickly early on and fine-tune stably later.

**Why include MPC if it needs no training?** MPC is the gold-standard planning baseline: it has access to the true current state and simulates future consequences. Showing that trained RL agents match or exceed it demonstrates that the learned policy has genuinely internalized useful structure rather than just memorizing common cases.
