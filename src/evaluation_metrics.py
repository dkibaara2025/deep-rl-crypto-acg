# Copyright (c) 2025.
# MIT License.

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - yhat)))


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def smape(y: np.ndarray, yhat: np.ndarray, eps: float = 1e-8) -> float:
    denom = (np.abs(y) + np.abs(yhat) + eps) / 2.0
    return float(np.mean(np.abs(y - yhat) / denom) * 100.0)


def directional_accuracy(y_prev: np.ndarray, y_true: np.ndarray, yhat: np.ndarray) -> float:
    s_true = np.sign(y_true - y_prev)
    s_pred = np.sign(yhat - y_prev)
    return float(np.mean((s_true == s_pred).astype(np.float32)))


def aggregate_by_horizon(per_arm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects per_arm_df with columns: ['asset','horizon','MAE','RMSE','sMAPE','DA']
    """
    return (
        per_arm_df.groupby("horizon")
        .agg(MAE=("MAE", "mean"), RMSE=("RMSE", "mean"), sMAPE=("sMAPE", "mean"), DA=("DA", "mean"))
        .reset_index()
        .sort_values("horizon")
    )


def summarize_metrics(y: np.ndarray, yhat: np.ndarray, yprev: np.ndarray) -> Dict[str, Any]:
    return {
        "MAE": mae(y, yhat),
        "RMSE": rmse(y, yhat),
        "sMAPE": smape(y, yhat),
        "DA": directional_accuracy(yprev, y, yhat),
    }
