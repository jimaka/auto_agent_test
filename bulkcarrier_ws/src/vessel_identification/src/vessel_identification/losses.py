"""Koopman training losses."""

from __future__ import annotations

import torch

from vessel_identification.models.koopman import DeepKoopman


def koopman_loss(
    model: DeepKoopman,
    x_seq: torch.Tensor,
    u_seq: torch.Tensor,
    one_step_weight: float = 1.0,
    multi_step_weight: float = 1.0,
    recon_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    x_seq: [B, N+1, nx] normalized
    u_seq: [B, N, nu] normalized
    """
    b, n_plus, _ = x_seq.shape
    n = n_plus - 1

    z_targets = model.multi_step_targets(x_seq)
    z0 = z_targets[:, 0, :]
    z_roll = model.rollout(z0, u_seq)

    loss_ms = torch.mean((z_roll - z_targets) ** 2)

    z0_enc = model.encode(x_seq[:, 0, :])
    z1_pred = model.step(z0_enc, u_seq[:, 0, :])
    z1_tgt = model.encode(x_seq[:, 1, :])
    loss_1 = torch.mean((z1_pred - z1_tgt) ** 2) + torch.mean((z0_enc - z_targets[:, 0, :]) ** 2)

    x_recon = model.decode(z_targets)
    loss_recon = torch.mean((x_recon - x_seq) ** 2)

    total = (
        multi_step_weight * loss_ms
        + one_step_weight * loss_1
        + recon_weight * loss_recon
    )
    metrics = {
        "loss": float(total.detach()),
        "loss_ms": float(loss_ms.detach()),
        "loss_1": float(loss_1.detach()),
        "loss_recon": float(loss_recon.detach()),
    }
    return total, metrics
