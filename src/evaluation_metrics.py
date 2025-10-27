"""Evaluation metrics for forecasting and bandit rewards."""
import numpy as np

def mae(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def directional_accuracy(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    sign_true = np.sign(np.diff(y_true))
    sign_pred = np.sign(np.diff(y_pred))
    return float(np.mean(sign_true == sign_pred))
