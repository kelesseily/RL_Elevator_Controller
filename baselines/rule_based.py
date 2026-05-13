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
        for _ in range(n_episodes):
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

    Decision logic (in priority order):
    1. If passengers inside need to get off at the current floor → STAY.
    2. Find the nearest pending call (hall or car call).
    3. Move toward it; if already there → STAY.
    4. If no calls exist → STAY.
    """

    name = "Nearest-Request"

    def select_action(self, env: ElevatorEnv) -> int:
        floor = env.elevator_floor

        # Priority 1: serve passengers already at this floor
        for p in env.passengers_inside:
            if p.destination == floor:
                return ACTION_STAY
        if env.waiting_passengers.get(floor):
            return ACTION_STAY

        # Collect all pending floor targets
        targets: list[int] = []
        for f in env.waiting_passengers:
            targets.append(f)
        for p in env.passengers_inside:
            targets.append(p.destination)

        if not targets:
            return ACTION_STAY

        nearest = min(targets, key=lambda f: abs(f - floor))
        if nearest > floor:
            return ACTION_UP
        elif nearest < floor:
            return ACTION_DOWN
        else:
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
