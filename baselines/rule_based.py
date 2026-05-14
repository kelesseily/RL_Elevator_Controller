"""
SmartLift - Baseline Controllers
==================================
Rule-based elevator controllers used as performance benchmarks.

Two strategies are implemented:
  1. NearestRequestController  – moves toward the nearest pending call (greedy)
  2. FCFSController             – serves requests in First-Come-First-Served order
"""

from __future__ import annotations
import numpy as np
from environment.elevator_env import (
    ElevatorEnv, ACTION_DOWN, ACTION_STAY, ACTION_UP,
    DIR_DOWN, DIR_IDLE, DIR_UP,
)

import copy

# ─────────────────────────────────────────────────────────────────────────────
# Base Interface
# ─────────────────────────────────────────────────────────────────────────────

class BaselineController:
    """Abstract baseline — subclasses implement `select_action`."""

    name: str = "Baseline"

    def select_action(self, env: ElevatorEnv) -> int:
        raise NotImplementedError

    def evaluate(self, env: ElevatorEnv, n_episodes: int = 50) -> dict:
        all_summaries = []
        for ep in range(n_episodes):
            # Re-seed each episode so baselines see same conditions as RL agents
            env.rng = np.random.default_rng(ep * 17 + 99)
            env.reset()
            done = False
            while not done:
                action = self.select_action(env)
                _, _, done, _ = env.step(action)
            all_summaries.append(env.get_episode_summary())

        keys = all_summaries[0].keys()
        return {k: float(np.mean([s[k] for s in all_summaries])) for k in keys}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Nearest-Request Controller
# ─────────────────────────────────────────────────────────────────────────────

class NearestRequestController(BaselineController):
    """
    Greedy nearest-floor heuristic.

    Decision logic:
    1. If anyone needs service at current floor → STAY.
    2. If carrying passengers → go to nearest destination.
    3. Otherwise → go to nearest waiting floor.
    4. No calls → STAY.
    """

    name = "Nearest-Request"

    def select_action(self, env: ElevatorEnv) -> int:
        floor = env.elevator_floor

        # Priority 1: always serve current floor if needed
        if any(p.destination == floor for p in env.passengers_inside):
            return ACTION_STAY
        if env.waiting_passengers.get(floor):
            return ACTION_STAY

        # Priority 2: deliver passengers inside first
        if env.passengers_inside:
            nearest_dest = min(
                (p.destination for p in env.passengers_inside),
                key=lambda f: abs(f - floor)
            )
            if nearest_dest > floor:
                return ACTION_UP
            elif nearest_dest < floor:
                return ACTION_DOWN
            return ACTION_STAY

        # Priority 3: go to nearest waiting floor
        if env.waiting_passengers:
            nearest_floor = min(
                env.waiting_passengers.keys(),
                key=lambda f: abs(f - floor)
            )
            if nearest_floor > floor:
                return ACTION_UP
            elif nearest_floor < floor:
                return ACTION_DOWN
            return ACTION_STAY

        return ACTION_STAY


# ─────────────────────────────────────────────────────────────────────────────
# 2. First-Come-First-Served (FCFS) Controller
# ─────────────────────────────────────────────────────────────────────────────

class FCFSController(BaselineController):
    """
    First-Come-First-Served controller.

    Tracks the oldest unanswered hall call and drives directly to it before
    attending to any newer calls.  Inside the car, it serves car calls in
    the order they were registered.
    """

    name = "FCFS"

    def __init__(self):
        self._target: int | None = None

    def select_action(self, env: ElevatorEnv) -> int:
        floor = env.elevator_floor

        # Always drop off / board at current floor first
        for p in env.passengers_inside:
            if p.destination == floor:
                self._target = None
                return ACTION_STAY
        if env.waiting_passengers.get(floor):
            self._target = None
            return ACTION_STAY

        # Determine target from oldest hall call or earliest car call
        oldest_hall_call: tuple[int, int] | None = None   # (arrival_time, floor)
        for f, plist in env.waiting_passengers.items():
            for p in plist:
                if oldest_hall_call is None or p.arrival_time < oldest_hall_call[0]:
                    oldest_hall_call = (p.arrival_time, f)

        earliest_car_call: tuple[int, int] | None = None  # (arrival_time, dest)
        for p in env.passengers_inside:
            if earliest_car_call is None or p.arrival_time < earliest_car_call[0]:
                earliest_car_call = (p.arrival_time, p.destination)

        # Choose target: hall call if no one inside, else car call
        if earliest_car_call:
            self._target = earliest_car_call[1]
        elif oldest_hall_call:
            self._target = oldest_hall_call[1]
        else:
            return ACTION_STAY

        if self._target > floor:
            return ACTION_UP
        elif self._target < floor:
            return ACTION_DOWN
        else:
            return ACTION_STAY
# ─────────────────────────────────────────────────────────────────────────────
# 3. Model Predictive Control (MPC) Controller
# ─────────────────────────────────────────────────────────────────────────────

class MPCController(BaselineController):
    """
    Model Predictive Control elevator controller.

    At each step, simulates all possible action sequences over a short
    horizon and picks the first action of the sequence that minimizes
    total predicted cost (waiting penalty + movement penalty).

    Unlike RL agents, MPC requires no training — it plans online using
    the environment's current state directly.
    """

    name = "MPC"

    def __init__(self, horizon: int = 6):
        self.horizon = horizon

    def select_action(self, env: ElevatorEnv) -> int:
        best_action = ACTION_STAY
        best_score  = float('-inf')

        for first_action in [ACTION_DOWN, ACTION_STAY, ACTION_UP]:
            score = self._rollout(env, first_action)
            if score > best_score:
                best_score  = score
                best_action = first_action

        return best_action

    def _rollout(self, env: ElevatorEnv, first_action: int) -> float:
        """
        Simulate horizon steps using a lightweight hand-coded model
        (no deepcopy — we track floor/passengers manually).
        No new arrivals are assumed during the horizon (conservative).
        """
        # Extract current state
        floor      = env.elevator_floor
        inside     = list(env.passengers_inside)          # list of Passenger
        waiting    = {f: list(ps) for f, ps in env.waiting_passengers.items() if ps}
        capacity   = env.max_capacity
        mv_penalty = env.movement_penalty
        wt_penalty = env.waiting_penalty
        svc_reward = env.service_reward

        total = 0.0
        discount = 1.0
        gamma = 0.95

        for step in range(self.horizon):
            action = first_action if step == 0 else self._greedy_action(floor, inside, waiting)

            # --- execute action ---
            moved = False
            if action == ACTION_UP and floor < env.n_floors - 1:
                floor += 1
                moved = True
            elif action == ACTION_DOWN and floor > 0:
                floor -= 1
                moved = True

            reward = -mv_penalty if moved else 0.0

            # --- serve floor: drop off ---
            still_riding = []
            served = 0
            for p in inside:
                if p.destination == floor:
                    served += 1
                else:
                    still_riding.append(p)
            inside = still_riding
            reward += served * svc_reward

            # --- serve floor: board ---
            if floor in waiting and waiting[floor]:
                space = capacity - len(inside)
                boarding = waiting[floor][:space]
                inside += boarding
                remaining = waiting[floor][space:]
                if remaining:
                    waiting[floor] = remaining
                else:
                    del waiting[floor]

            # --- waiting penalty (capped) ---
            total_waiting = sum(len(ps) for ps in waiting.values())
            reward -= wt_penalty * min(total_waiting, 10)

            total += discount * reward
            discount *= gamma

        return total

    def _greedy_action(self, floor, inside, waiting) -> int:
        """Nearest-request heuristic for steps 2..horizon."""
        # Serve current floor first
        for p in inside:
            if p.destination == floor:
                return ACTION_STAY
        if waiting.get(floor):
            return ACTION_STAY

        targets = list(waiting.keys()) + [p.destination for p in inside]
        if not targets:
            return ACTION_STAY

        nearest = min(targets, key=lambda f: abs(f - floor))
        if nearest > floor:
            return ACTION_UP
        elif nearest < floor:
            return ACTION_DOWN
        return ACTION_STAY