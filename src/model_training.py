# Copyright (c) 2025.
# MIT License.

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .evaluation_metrics import mae, rmse, smape, directional_accuracy, aggregate_by_horizon
from .curriculum_bandit import UCB1Teacher, UCB1Config


# ---------- Repro ----------
def set_seed(s: int) -> None:
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ---------- Model ----------
class PolicyLSTM(nn.Module):
    """
    Minimal Gaussian policy forecaster with LSTM encoder.
    Returns (mu, log_sigma) where sigma is a learned global scale.
    """
    def __init__(self, input_dim: int = 1, hidden: int = 64, layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=layers,
            dropout=(dropout if layers > 1 else 0.0),
            batch_first=True,
        )
        self.mu = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.log_sigma = nn.Parameter(torch.tensor(-0.5))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        o, _ = self.lstm(x)     # (B, T, H)
        last = o[:, -1, :]      # (B, H)
        mu = self.mu(last)      # (B, 1)
        mu = torch.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
        log_sigma = self.log_sigma.expand_as(mu)
        return mu, log_sigma


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------- Mini-batching & Eval ----------
def sample_minibatch(
    data_xy: Dict[Tuple[str, str, int], Tuple[np.ndarray, np.ndarray, np.ndarray]],
    arms: List[Tuple[str, int]],
    arm_idx: int,
    bs: int,
    device: torch.device,
):
    a, h = arms[arm_idx]
    X, Y, P = data_xy[(a, "train", h)]
    n = len(Y)
    if n == 0:
        idx = np.array([0])
    else:
        idx = np.random.randint(0, n, size=(min(bs, n),))
    xb = torch.from_numpy(X[idx]).float().to(device)
    yb = torch.from_numpy(Y[idx]).float().to(device)
    pb = torch.from_numpy(P[idx]).float().to(device)
    return xb, yb, pb


def eval_val_mse(
    model: nn.Module,
    data_xy: Dict[Tuple[str, str, int], Tuple[np.ndarray, np.ndarray, np.ndarray]],
    arms: List[Tuple[str, int]],
    arm_idx: int,
    device: torch.device,
) -> float:
    a, h = arms[arm_idx]
    X, Y, _ = data_xy[(a, "val", h)]
    with torch.no_grad():
        xb = torch.from_numpy(X).float().to(device)
        mu, _ = model(xb)
        yhat = mu.squeeze(1).cpu().numpy()
    return float(np.mean((Y.squeeze(1) - yhat) ** 2))


def eval_overall_val_mse(
    model: nn.Module,
    data_xy: Dict[Tuple[str, str, int], Tuple[np.ndarray, np.ndarray, np.ndarray]],
    arms: List[Tuple[str, int]],
    device: torch.device,
) -> float:
    vals = [eval_val_mse(model, data_xy, arms, i, device) for i, _ in enumerate(arms)]
    return float(np.mean(vals)) if len(vals) else np.nan


def evaluate_test(
    model: nn.Module,
    data_xy: Dict[Tuple[str, str, int], Tuple[np.ndarray, np.ndarray, np.ndarray]],
    arms: List[Tuple[str, int]],
    device: torch.device,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for a, h in arms:
        Xte, Yte, Prev = data_xy[(a, "test", h)]
        with torch.no_grad():
            xb = torch.from_numpy(Xte).float().to(device)
            mu, _ = model(xb)
            yhat = mu.squeeze(1).cpu().numpy()
        y = Yte.squeeze(1)
        p = Prev.squeeze(1)
        rows.append(
            dict(
                asset=a,
                horizon=h,
                MAE=mae(y, yhat),
                RMSE=rmse(y, yhat),
                sMAPE=smape(y, yhat),
                DA=directional_accuracy(p, y, yhat),
            )
        )
    per_arm = pd.DataFrame(rows).sort_values(["horizon", "asset"])
    agg = aggregate_by_horizon(per_arm)
    return per_arm, agg


# ---------- Bandit Training ----------
@dataclass
class TrainConfig:
    seed: int = 1337
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    hidden: int = 64
    layers: int = 1
    dropout: float = 0.0
    lr: float = 2e-3
    entropy_beta: float = 1e-3
    aux_sup_weight: float = 1e-1
    batch_size: int = 256
    rounds: int = 150
    steps_per_pull: int = 1
    # early stopping across arms (global val MSE)
    patience_rounds: int = 15
    eval_every: int = 10
    ucb_c: float = 1.2


def bandit_run_once(
    df_feat: pd.DataFrame,
    data_xy: Dict[Tuple[str, str, int], Tuple[np.ndarray, np.ndarray, np.ndarray]],
    arms: List[Tuple[str, int]],
    cfg: TrainConfig,
):
    """
    One run of REINFORCE + auxiliary MSE under UCB1 teacher.
    Returns (per_arm_results_df, agg_df, model, rounds_used)
    """
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    input_dim = len([c for c in ["z", "dz", "vol14", "rsi14", "macd_z", "dist_ma20_z"] if c in df_feat.columns])

    model = PolicyLSTM(input_dim=input_dim, hidden=cfg.hidden, layers=cfg.layers, dropout=cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    teacher = UCB1Teacher(len(arms), UCB1Config(c=cfg.ucb_c))

    def reinforce_step(xb: torch.Tensor, yb: torch.Tensor):
        model.train()
        opt.zero_grad()
        mu, log_sigma = model(xb)
        sigma = torch.exp(log_sigma)
        dist = torch.distributions.Normal(mu, sigma)
        a = dist.rsample()
        r = -((a - yb) ** 2)

        # simple moving baseline to reduce variance
        reinforce_step.baseline = (
            0.9 * reinforce_step.baseline + 0.1 * r.mean().detach()
            if hasattr(reinforce_step, "baseline")
            else r.mean().detach()
        )
        adv = r - reinforce_step.baseline
        loss = -(dist.log_prob(a) * adv.detach()).mean()
        loss += -cfg.entropy_beta * dist.entropy().mean()
        loss += cfg.aux_sup_weight * nn.MSELoss()(mu, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    # warm start
    for i in range(min(len(arms), 3)):
        xb, yb, _ = sample_minibatch(data_xy, arms, i, cfg.batch_size, device)
        reinforce_step(xb, yb)

    best = (np.inf, None, 0)  # (global_val, state_dict, round)
    no_improve = 0
    for t in range(1, cfg.rounds + 1):
        i = teacher.select()
        vb = eval_val_mse(model, data_xy, arms, i, device)
        for _ in range(cfg.steps_per_pull):
            xb, yb, _ = sample_minibatch(data_xy, arms, i, cfg.batch_size, device)
            reinforce_step(xb, yb)
        va = eval_val_mse(model, data_xy, arms, i, device)
        teacher.update(i, float(vb - va))

        if t % cfg.eval_every == 0:
            gval = eval_overall_val_mse(model, data_xy, arms, device)
            if gval < best[0]:
                best = (gval, {k: v.cpu().clone() for k, v in model.state_dict().items()}, t)
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= cfg.patience_rounds:
                break

    rounds_used = best[2] if best[1] is not None else cfg.rounds
    if best[1] is not None:
        model.load_state_dict(best[1])

    per_arm, agg = evaluate_test(model, data_xy, arms, device)
    return per_arm, agg, model, rounds_used


# ---------- Supervised Baseline ----------
@dataclass
class SupervisedConfig:
    seed: int = 1337
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    hidden: int = 64
    layers: int = 1
    dropout: float = 0.0
    lr: float = 2e-3
    batch_size: int = 256
    epochs: int = 4


def supervised_run(
    df_feat: pd.DataFrame,
    data_xy: Dict[Tuple[str, str, int], Tuple[np.ndarray, np.ndarray, np.ndarray]],
    arms: List[Tuple[str, int]],
    cfg: SupervisedConfig,
):
    """
    Pooled supervised training (pure MSE) across all arms.
    Returns (per_arm_df, agg_df, model)
    """
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    input_dim = len([c for c in ["z", "dz", "vol14", "rsi14", "macd_z", "dist_ma20_z"] if c in df_feat.columns])

    model = PolicyLSTM(input_dim=input_dim, hidden=cfg.hidden, layers=cfg.layers, dropout=cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    # pool train/val
    Xtr, Ytr = [], []
    Xva, Yva = [], []
    for a, h in arms:
        Xt, Yt, _ = data_xy[(a, "train", h)]
        Xv, Yv, _ = data_xy[(a, "val", h)]
        Xtr.append(Xt); Ytr.append(Yt)
        Xva.append(Xv); Yva.append(Yv)

    Xtr = torch.from_numpy(np.concatenate(Xtr, axis=0)).float().to(device)
    Ytr = torch.from_numpy(np.concatenate(Ytr, axis=0)).float().to(device)
    Xva = torch.from_numpy(np.concatenate(Xva, axis=0)).float().to(device)
    Yva = torch.from_numpy(np.concatenate(Yva, axis=0)).float().to(device)

    best = (np.inf, None)
    for _ in range(cfg.epochs):
        model.train()
        idx = torch.randperm(Xtr.shape[0])
        for start in range(0, len(idx), cfg.batch_size):
            sel = idx[start : start + cfg.batch_size]
            xb, yb = Xtr[sel], Ytr[sel]
            opt.zero_grad()
            mu, _ = model(xb)
            loss = loss_fn(mu, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            mu, _ = model(Xva)
            va = loss_fn(mu, Yva).item()
        if va < best[0]:
            best = (va, {k: v.cpu().clone() for k, v in model.state_dict().items()})

    if best[1] is not None:
        model.load_state_dict(best[1])

    per_arm, agg = evaluate_test(model, data_xy, arms, device)
    return per_arm, agg, model
