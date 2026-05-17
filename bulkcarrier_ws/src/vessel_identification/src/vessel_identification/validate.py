"""N-step horizon validation and tube tightening export."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from vessel_identification.config import KoopmanConfig
from vessel_identification.dataset import load_npz_split
from vessel_identification.models.koopman import DeepKoopman
from vessel_identification.normalizer import Normalizer
from vessel_identification.trainer import load_checkpoint


def horizon_metrics(
    model: DeepKoopman,
    norm_x: Normalizer,
    norm_u: Normalizer,
    x: np.ndarray,
    u: np.ndarray,
    horizon: int,
    device: torch.device,
) -> Dict[str, float | np.ndarray]:
    model.eval()
    n_win = len(x) - horizon
    if n_win < 1:
        raise ValueError("Series too short for horizon validation")

    z_rmse_steps = []
    x_rmse_steps = []

    with torch.no_grad():
        for i in range(0, n_win, max(1, n_win // 500)):
            x_win = norm_x.normalize(x[i : i + horizon + 1])
            u_win = norm_u.normalize(u[i : i + horizon])
            x_t = torch.from_numpy(x_win).float().unsqueeze(0).to(device)
            u_t = torch.from_numpy(u_win).float().unsqueeze(0).to(device)

            z_tgt = model.multi_step_targets(x_t)
            z_roll = model.rollout(z_tgt[:, 0, :], u_t)
            z_err = (z_roll - z_tgt).squeeze(0).cpu().numpy()
            z_rmse_steps.append(np.sqrt(np.mean(z_err**2, axis=1)))

            x_pred = model.decode(z_roll).squeeze(0).cpu().numpy()
            x_err = x_pred - x_t.squeeze(0).cpu().numpy()
            x_rmse_steps.append(np.sqrt(np.mean(x_err**2, axis=1)))

    z_per_step = np.mean(np.stack(z_rmse_steps), axis=0)
    x_per_step = np.mean(np.stack(x_rmse_steps), axis=0)

    return {
        "n_step_z_rmse": float(np.mean(z_per_step)),
        "n_step_x_rmse": float(np.mean(x_per_step)),
        "z_rmse_per_step": z_per_step,
        "x_rmse_per_step": x_per_step,
    }


def write_tube_csv(path: Path, x_rmse_per_step: np.ndarray, quantile: float = 0.95) -> None:
    """Map state RMSE envelope to conservative tightening placeholders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "eps_cross", "eps_psi", "eps_u"])
        for i, rmse in enumerate(x_rmse_per_step[1:], start=1):
            # Heuristic: scale total RMSE into channel-wise eps
            eps_cross = float(rmse * 2.0 * quantile)
            eps_psi = float(rmse * 0.2 * quantile)
            eps_u = float(rmse * 0.3 * quantile)
            w.writerow([i, eps_cross, eps_psi, eps_u])


def validate_checkpoint(
    checkpoint: Path,
    data_dir: Path,
    split: str = "val",
    horizon: int | None = None,
) -> Tuple[Dict, DeepKoopman, KoopmanConfig]:
    device = torch.device("cpu")
    model, cfg, norm_x, norm_u, _ = load_checkpoint(checkpoint, device)
    horizon = horizon or cfg.horizon_N
    arrays = load_npz_split(data_dir, split)
    metrics = horizon_metrics(
        model, norm_x, norm_u, arrays["x"], arrays["u"], horizon, device
    )
    summary = {
        "n_step_z_rmse": metrics["n_step_z_rmse"],
        "n_step_x_rmse": metrics["n_step_x_rmse"],
        "horizon_N": horizon,
        "split": split,
    }
    return summary, metrics, model, cfg
