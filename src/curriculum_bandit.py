# Copyright (c) 2025.
# MIT License.

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, asdict


@dataclass
class UCB1Config:
    c: float = 1.2  # exploration constant

    def to_dict(self):  # handy for logging
        return asdict(self)


class UCB1Teacher:
    """
    Classic UCB1 scheduler over K arms.
    Keeps incremental means and pull counts; returns arm indices.
    """
    def __init__(self, n_arms: int, cfg: UCB1Config | None = None):
        self.cfg = cfg or UCB1Config()
        self.n = np.zeros(n_arms, dtype=np.int64)       # pulls per arm
        self.mean = np.zeros(n_arms, dtype=np.float64)  # mean reward per arm
        self.t = 0                                      # total pulls

    @property
    def n_arms(self) -> int:
        return len(self.n)

    def select(self) -> int:
        self.t += 1
        # Pull each arm at least once
        for i in range(self.n_arms):
            if self.n[i] == 0:
                return i
        ucb = self.mean + self.cfg.c * np.sqrt(2.0 * math.log(self.t) / self.n)
        return int(np.argmax(ucb))

    def update(self, i: int, reward: float) -> None:
        self.n[i] += 1
        self.mean[i] += (reward - self.mean[i]) / self.n[i]


class UniformTeacher:
    """
    Baseline scheduler: chooses arms uniformly at random.
    """
    def __init__(self, n_arms: int, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.n = np.zeros(n_arms, dtype=np.int64)
        self.mean = np.zeros(n_arms, dtype=np.float64)
        self.t = 0

    def select(self) -> int:
        self.t += 1
        return int(self.rng.integers(0, len(self.n)))

    def update(self, i: int, reward: float) -> None:
        self.n[i] += 1
        self.mean[i] += (reward - self.mean[i]) / self.n[i]
