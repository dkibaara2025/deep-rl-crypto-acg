"""Bandit-based teacher policies for Automatic Curriculum Generation (ACG)."""
from dataclasses import dataclass
import numpy as np

@dataclass
class UCBAgent:
    n_arms: int
    counts: np.ndarray = None
    values: np.ndarray = None
    t: int = 0
    c: float = 2.0

    def __post_init__(self):
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.values = np.zeros(self.n_arms, dtype=float)

    def select_arm(self):
        self.t += 1
        ucb = self.values + self.c * np.sqrt(np.log(self.t + 1) / (self.counts + 1e-9))
        return int(np.argmax(ucb))

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        self.values[arm] = value + (reward - value) / n


@dataclass
class ThompsonSamplingAgent:
    n_arms: int
    alpha: np.ndarray = None
    beta: np.ndarray = None

    def __post_init__(self):
        self.alpha = np.ones(self.n_arms)
        self.beta = np.ones(self.n_arms)

    def select_arm(self):
        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float):
        # reward assumed in [0,1]
        self.alpha[arm] += reward
        self.beta[arm] += 1 - reward
