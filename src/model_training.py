"""Minimal training stubs for student models (e.g., KAN, LSTM, Transformer baselines)."""
import numpy as np

class DummyStudent:
    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full((len(X),), getattr(self, 'mean_', 0.0))
