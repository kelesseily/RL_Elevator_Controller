"""
SmartLift - Elevator Environment (MDP Formulation)
====================================================
Models a multi-floor elevator system as a Markov Decision Process (MDP).

State  : (elevator_floor, direction, hall_calls_up, hall_calls_down, car_calls, passengers_in_elevator)
Action : 0 = Move Down, 1 = Stay/Serve, 2 = Move Up
Reward : Penalizes waiting time and unnecessary movement; rewards passenger service.
"""

import numpy as np
import random
from collections import defaultdict


# ─── Constants ───────────────────────────────────────────────────────────────
ACTION_DOWN  = 0
ACTION_STAY  = 1
ACTION_UP    = 2
ACTION_NAMES = {ACTION_DOWN: "DOWN", ACTION_STAY: "STAY", ACTION_UP: "UP"}

DIR_DOWN = -1
DIR_IDLE = 0
DIR_UP   = 1


class Passenger:
    """Represents a single passenger in the system."""
    def __init__(self, origin: int, destination: int, arrival_time: int):
        self.origin        = origin
        self.destination   = destination
        self.arrival_time  = arrival_time
        self.pickup_time   = None
        self.dropoff_time  = None

    @property
    def waiting_time(self) -> int:
        if self.pickup_time is None:
            return 0
        return self.pickup_time - self.arrival_time

    @property
    def ride_time(self) -> int:
        if self.dropoff_time is None or self.pickup_time is None:
            return 0
        return self.dropoff_time - self.pickup_time

    @property
    def total_time(self) -> int:
        if self.dropoff_time is None:
            return 0
        return self.dropoff_time - self.arrival_time


class ElevatorEnv:
    """
    Custom elevator simulation environment.

    Parameters
    ----------
    n_floors         : number of floors in the building
    max_capacity     : maximum passengers the elevator can carry
    arrival_rate     : Poisson arrival rate (passengers per step)
    max_steps        : episode length
    seed             : random seed for reproducibility
    movement_penalty : reward penalty per elevator movement step
    waiting_penalty  : reward penalty per passenger per waiting step
    service_reward   : reward given when a passenger is delivered
    """

    def __init__(
        self,
        n_floors: int          = 10,
        max_capacity: int      = 8,
        arrival_rate: float    = 0.3,
        max_steps: int         = 500,
        seed: int              = 42,
        movement_penalty: float = 0.5,
        waiting_penalty: float  = 1.0,
        service_reward: float   = 10.0,
    ):
        self.n_floors          = n_floors
        self.max_capacity      = max_capacity
        self.arrival_rate      = arrival_rate
        self.max_steps         = max_steps
        self.movement_penalty  = movement_penalty
        self.waiting_penalty   = waiting_penalty
        self.service_reward    = service_reward

        self.rng = np.random.default_rng(seed)
        random.seed(seed)

        # Action / observation spaces (discrete)
        self.n_actions = 3                    # DOWN, STAY, UP
        self.action_space = list(range(self.n_actions))

        self.reset()

    # ──────────────────────────────────────────────────────────────────────────
    # Core MDP Interface
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self) -> tuple:
        """Reset environment and return initial state."""
        self.current_step       = 0
        self.elevator_floor     = self.n_floors // 2   # start in the middle
        self.elevator_direction = DIR_IDLE
        self.passengers_inside  : list[Passenger] = []   # riding the elevator
        self.waiting_passengers : dict[int, list[Passenger]] = defaultdict(list)  # floor → list
        self.served_passengers  : list[Passenger] = []
        self.total_movements    = 0

        self.episode_metrics = {
            "waiting_times"   : [],
            "total_times"     : [],
            "rewards"         : [],
            "movements"       : 0,
            "served"          : 0,
        }

        return self._get_state()

    def step(self, action: int) -> tuple:
        """
        Apply action, advance simulation one time step.

        Returns
        -------
        next_state, reward, done, info
        """
        assert action in self.action_space, f"Invalid action: {action}"

        reward = 0.0

        # 1. Spawn new passengers (stochastic arrivals)
        self._spawn_passengers()

        # 2. Execute action
        moved = self._execute_action(action)
        if moved:
            reward -= self.movement_penalty
            self.total_movements += 1
            self.episode_metrics["movements"] += 1

        # 3. Serve passengers at current floor (board + alight)
        served_count = self._serve_floor()
        reward += served_count * self.service_reward
        self.episode_metrics["served"] += served_count

        # 4. Waiting penalty — capped at 10 to prevent unbounded explosion
        total_waiting = sum(len(p) for p in self.waiting_passengers.values())
        reward -= self.waiting_penalty * min(total_waiting, 10)

        # 5. Record
        self.episode_metrics["rewards"].append(reward)
        self.current_step += 1
        done = self.current_step >= self.max_steps

        next_state = self._get_state()
        total_waiting = sum(
            len(plist) for plist in self.waiting_passengers.values()
        )
        total_waiting += len(self.passengers_inside)
        info = {
            "floor"            : self.elevator_floor,
            "direction"        : self.elevator_direction,
            "passengers_inside": len(self.passengers_inside),
            "waiting_total"    : total_waiting,
            "served_this_step" : served_count,
        }

        return next_state, reward, done, info

    # ──────────────────────────────────────────────────────────────────────────
    # State Encoding
    # ──────────────────────────────────────────────────────────────────────────

    def _get_state(self) -> tuple:
        """
        Returns a discrete, hashable state tuple:
        (elevator_floor,
         direction_index,          # 0=down, 1=idle, 2=up
         hall_calls_up_bitmask,    # integer bitmask over floors
         hall_calls_down_bitmask,
         car_calls_bitmask,        # destination buttons pressed inside car
         passengers_in_elevator_bin)  # bucketed count
        """
        direction_idx = self.elevator_direction + 1   # map -1,0,1 → 0,1,2

        floor = self.elevator_floor

        # Binary signals only — keeps state space small and learnable
        calls_above = int(any(f > floor for f in self.waiting_passengers))
        calls_below = int(any(f < floor for f in self.waiting_passengers))
        calls_here  = int(floor in self.waiting_passengers and
                         bool(self.waiting_passengers.get(floor)))

        dest_above  = int(any(p.destination > floor for p in self.passengers_inside))
        dest_below  = int(any(p.destination < floor for p in self.passengers_inside))

        # 0 = empty, 1 = partial, 2 = full
        n_inside = len(self.passengers_inside)
        if n_inside == 0:
            load = 0
        elif n_inside < self.max_capacity:
            load = 1
        else:
            load = 2

        return (
            floor,
            direction_idx,
            calls_above,
            calls_below,
            calls_here,
            dest_above,
            dest_below,
            load,
        )

    def get_state_as_dict(self) -> dict:
        """Human-readable state (for debugging / visualisation)."""
        waiting_by_floor = {
            f: len(plist)
            for f, plist in self.waiting_passengers.items()
            if plist
        }
        return {
            "floor"           : self.elevator_floor,
            "direction"       : {-1: "DOWN", 0: "IDLE", 1: "UP"}[self.elevator_direction],
            "inside"          : len(self.passengers_inside),
            "waiting_by_floor": waiting_by_floor,
            "step"            : self.current_step,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Internal Mechanics
    # ──────────────────────────────────────────────────────────────────────────

    def _spawn_passengers(self):
        """Poisson-distributed passenger arrivals each time step."""
        n_arrivals = self.rng.poisson(self.arrival_rate)
        for _ in range(n_arrivals):
            origin = int(self.rng.integers(0, self.n_floors))
            # Destination uniformly sampled from all OTHER floors
            destination = origin
            while destination == origin:
                destination = int(self.rng.integers(0, self.n_floors))
            p = Passenger(origin, destination, self.current_step)
            self.waiting_passengers[origin].append(p)

    def _execute_action(self, action: int) -> bool:
        """Move elevator. Returns True if a physical movement was made."""
        if action == ACTION_UP and self.elevator_floor < self.n_floors - 1:
            self.elevator_floor     += 1
            self.elevator_direction  = DIR_UP
            return True
        elif action == ACTION_DOWN and self.elevator_floor > 0:
            self.elevator_floor     -= 1
            self.elevator_direction  = DIR_DOWN
            return True
        else:
            self.elevator_direction = DIR_IDLE
            return False

    def _serve_floor(self) -> int:
        """
        Drop off arriving passengers and pick up waiting passengers.
        
        Returns
        -------
        served_count: number of passengers delivered
        """
        served = 0
        current_floor = self.elevator_floor

        # Drop off passengers whose destination is this floor
        still_riding = []
        for p in self.passengers_inside:
            if p.destination == current_floor:
                p.dropoff_time = self.current_step
                self.served_passengers.append(p)
                self.episode_metrics["waiting_times"].append(p.waiting_time)
                self.episode_metrics["total_times"].append(p.total_time)
                served += 1
            else:
                still_riding.append(p)
        self.passengers_inside = still_riding

        # Board waiting passengers (up to capacity)
        if self.waiting_passengers.get(current_floor):
            available_space = self.max_capacity - len(self.passengers_inside)
            boarding = self.waiting_passengers[current_floor][:available_space]
            for p in boarding:
                p.pickup_time = self.current_step
                self.passengers_inside.append(p)
            remaining = self.waiting_passengers[current_floor][available_space:]
            if remaining:
                self.waiting_passengers[current_floor] = remaining
            else:
                del self.waiting_passengers[current_floor]

        return served

    # ──────────────────────────────────────────────────────────────────────────
    # Episode Summary
    # ──────────────────────────────────────────────────────────────────────────

    def get_episode_summary(self) -> dict:
        wt = self.episode_metrics["waiting_times"]
        tt = self.episode_metrics["total_times"]
        return {
            "total_served"      : self.episode_metrics["served"],
            "total_movements"   : self.episode_metrics["movements"],
            "avg_waiting_time"  : np.mean(wt) if wt else 0.0,
            "max_waiting_time"  : np.max(wt)  if wt else 0.0,
            "avg_total_time"    : np.mean(tt) if tt else 0.0,
            "total_reward"      : sum(self.episode_metrics["rewards"]),
        }

    def has_pending_calls(self) -> bool:
        return bool(self.waiting_passengers) or bool(self.passengers_inside)

    def __repr__(self):
        s = self.get_state_as_dict()
        return (f"ElevatorEnv(floor={s['floor']}, dir={s['direction']}, "
                f"inside={s['inside']}, waiting={s['waiting_by_floor']})")
