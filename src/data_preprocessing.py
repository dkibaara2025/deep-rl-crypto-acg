# Copyright (c) 2025.
# MIT License.

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd


# ---------- Config ----------
@dataclass
class FeatureConfig:
    win: int = 64
    horizons: Tuple[int, ...] = (1, 3, 7)
    assets_max: int = 5
    # Sanitization
    clip_val: float = 10.0


# ---------- IO & Normalization ----------
REQUIRED_COLS = {"timestamp", "symbol", "close", "split"}

CANONICAL_RENAMES = {
    "ticker": "symbol",
    "asset": "symbol",
    "price": "close",
    "close_price": "close",
}


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    # normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    for k, v in CANONICAL_RENAMES.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})
    if not REQUIRED_COLS.issubset(df.columns):
        raise ValueError(f"Dataset missing columns: {REQUIRED_COLS - set(df.columns)}")

    # numeric coercion for close
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df


def top_assets(df: pd.DataFrame, k: int) -> List[str]:
    return (
        df.groupby("symbol")
        .size()
        .sort_values(ascending=False)
        .head(k)
        .index.tolist()
    )


def fit_close_scalers(df: pd.DataFrame, assets: List[str]) -> Dict[str, Tuple[float, float]]:
    scalers: Dict[str, Tuple[float, float]] = {}
    for a, g in df[(df["split"] == "train") & (df["symbol"].isin(assets))].groupby("symbol"):
        mu = pd.to_numeric(g["close"], errors="coerce").mean()
        sd = pd.to_numeric(g["close"], errors="coerce").std(ddof=0)
        if not np.isfinite(sd) or sd <= 0:
            sd = 1.0
        mu = 0.0 if not np.isfinite(mu) else float(mu)
        scalers[a] = (float(mu), float(sd))
    return scalers


# ---------- Feature Engineering ----------
def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.rolling(period, min_periods=period).mean()
    roll_down = down.rolling(period, min_periods=period).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def build_features(
    df_raw: pd.DataFrame,
    assets: List[str],
    augmented: bool,
    clip_val: float = 10.0,
) -> pd.DataFrame:
    """
    Returns dataframe with:
      - columns: timestamp, symbol, split, z, [dz, vol14, rsi14, macd_z, dist_ma20_z] if augmented
      - numeric coercion + sanitize (inf->NaN->0, clip)
    """
    df = df_raw[df_raw["symbol"].isin(assets)].sort_values(["symbol", "timestamp"]).copy()
    scalers = fit_close_scalers(df, assets)

    mu_map = {k: v[0] for k, v in scalers.items()}
    sd_map = {k: v[1] for k, v in scalers.items()}
    df["z"] = (df["close"] - df["symbol"].map(mu_map)) / df["symbol"].map(sd_map)
    df["z"] = pd.to_numeric(df["z"], errors="coerce")

    if not augmented:
        out = df[["timestamp", "symbol", "split", "z"]].copy()
        out["z"] = out["z"].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-clip_val, clip_val)
        return out

    # augmented
    feats = []
    for a, g in df.groupby("symbol"):
        g = g.sort_values("timestamp").copy()
        z = pd.to_numeric(g["z"], errors="coerce")
        dz = z.diff()
        vol14 = dz.rolling(14, min_periods=14).std()
        rsi14 = _rsi(g["close"], 14)
        ma5 = z.rolling(5, min_periods=5).mean()
        ma20 = z.rolling(20, min_periods=20).mean()
        macd_z = ma5 - ma20
        dist_ma20_z = z - ma20

        tmp = pd.DataFrame(
            {
                "timestamp": g["timestamp"].values,
                "symbol": a,
                "split": g["split"].values,
                "z": z.values,
                "dz": dz.values,
                "vol14": vol14.values,
                "rsi14": rsi14.values,
                "macd_z": macd_z.values,
                "dist_ma20_z": dist_ma20_z.values,
            }
        )

        std_cols = ["dz", "vol14", "rsi14", "macd_z", "dist_ma20_z"]
        for c in ["z"] + std_cols:
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

        train_mask = tmp["split"] == "train"
        for c in std_cols:
            mu = pd.to_numeric(tmp.loc[train_mask, c], errors="coerce").mean()
            sd = pd.to_numeric(tmp.loc[train_mask, c], errors="coerce").std(ddof=0)
            if not np.isfinite(sd) or sd <= 0:
                sd = 1.0
            mu = 0.0 if not np.isfinite(mu) else float(mu)
            tmp[c] = (tmp[c] - mu) / sd

        tmp[["z"] + std_cols] = (
            tmp[["z"] + std_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-clip_val, clip_val)
        )
        feats.append(tmp)

    out = pd.concat(feats, ignore_index=True)
    for c in ["z", "dz", "vol14", "rsi14", "macd_z", "dist_ma20_z"]:
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out


# ---------- Windowing ----------
def make_xy_windows(
    g: pd.DataFrame, feature_cols: List[str], horizon: int, win: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | Tuple[None, None, None]:
    vals = g[feature_cols].values.astype(np.float32)
    z = g["z"].values.astype(np.float32)

    X, Y, Prev = [], [], []
    dropped = 0
    for t in range(win - 1, len(vals) - horizon):
        x = vals[t - win + 1 : t + 1, :]
        y = z[t + horizon]
        p = z[t]
        if not (np.isfinite(x).all() and np.isfinite(y) and np.isfinite(p)):
            dropped += 1
            continue
        X.append(x)
        Y.append([y])
        Prev.append([p])

    if dropped > 0:
        print(f"[make_xy_windows] Dropped {dropped} windows (win={win}, h={horizon}).")
    if not X:
        return None, None, None
    return np.stack(X), np.stack(Y), np.stack(Prev)


def build_arm_data(
    df_feat: pd.DataFrame, horizons: Tuple[int, ...], win: int
) -> Tuple[Dict[Tuple[str, str, int], Tuple[np.ndarray, np.ndarray, np.ndarray]], List[Tuple[str, int]], pd.DataFrame]:
    all_cols = ["z", "dz", "vol14", "rsi14", "macd_z", "dist_ma20_z"]
    feature_cols = [c for c in df_feat.columns if c in all_cols]
    data_xy: Dict[Tuple[str, str, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    arms: List[Tuple[str, int]] = []
    coverage = []

    for a, g in df_feat.groupby("symbol"):
        g = g.sort_values("timestamp")
        for split in ["train", "val", "test"]:
            gs = g[g["split"] == split].reset_index(drop=True)
            for h in horizons:
                X, Y, P = make_xy_windows(gs, feature_cols, h, win)
                if X is not None:
                    data_xy[(a, split, h)] = (X, Y, P)
                    coverage.append((a, split, h, len(Y)))
        for h in horizons:
            ok = all(
                ((a, sp, h) in data_xy and data_xy[(a, sp, h)][1].shape[0] > 0)
                for sp in ["train", "val", "test"]
            )
            if ok:
                arms.append((a, h))

    cov_df = pd.DataFrame(coverage, columns=["asset", "split", "horizon", "samples"])
    return data_xy, arms, cov_df


def downsample_train(
    data_xy: Dict[Tuple[str, str, int], Tuple[np.ndarray, np.ndarray, np.ndarray]],
    frac: float,
    rng_seed: int = 123,
) -> Dict[Tuple[str, str, int], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if frac >= 0.999:
        return data_xy
    out: Dict[Tuple[str, str, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    rng = np.random.default_rng(rng_seed)
    for k in list(data_xy.keys()):
        a, split, h = k
        X, Y, P = data_xy[k]
        if split == "train":
            n = len(Y)
            m = max(1, int(round(n * frac)))
            idx = rng.choice(n, size=m, replace=False)
            out[k] = (X[idx], Y[idx], P[idx])
        else:
            out[k] = (X, Y, P)
    return out
