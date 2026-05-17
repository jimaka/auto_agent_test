"""Training loop and checkpointing."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from vessel_identification.config import KoopmanConfig
from vessel_identification.dataset import KoopmanSequenceDataset, build_dataloaders
from vessel_identification.losses import koopman_loss
from vessel_identification.models.koopman import DeepKoopman
from vessel_identification.normalizer import Normalizer


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(
    cfg: KoopmanConfig,
    data_dir: Path,
    out_dir: Path,
) -> Path:
    set_seed(cfg.training.seed)
    device = resolve_device(cfg.training.device)
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f)

    datasets, norm_x, norm_u = build_dataloaders(
        data_dir, cfg.horizon_N, cfg.training.batch_size
    )
    if "val" not in datasets:
        datasets["val"] = datasets["train"]
    norm_x.save(out_dir / "norm_x.json")
    norm_u.save(out_dir / "norm_u.json")

    loaders = {
        k: DataLoader(
            ds,
            batch_size=cfg.training.batch_size,
            shuffle=(k == "train"),
            drop_last=(k == "train"),
            num_workers=0,
        )
        for k, ds in datasets.items()
    }

    model = DeepKoopman(
        nx=cfg.nx,
        nu=cfg.nu,
        nz=cfg.nz,
        hidden=cfg.encoder.hidden,
        activation=cfg.encoder.activation,
    ).to(device)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    best_val = float("inf")
    best_path = out_dir / "best.pt"
    history = []

    for epoch in range(1, cfg.training.epochs + 1):
        tr_loss = _run_epoch(
            model, loaders.get("train"), opt, device, cfg, train=True
        )
        val_loss = _run_epoch(
            model, loaders.get("val"), None, device, cfg, train=False
        )
        row = {"epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss}
        history.append(row)
        if val_loss < best_val:
            best_val = val_loss
            _save_checkpoint(best_path, model, norm_x, norm_u, cfg, epoch, best_val)

        if epoch % max(1, cfg.training.epochs // 10) == 0 or epoch == 1:
            print(
                f"epoch {epoch}/{cfg.training.epochs} "
                f"train={tr_loss:.6f} val={val_loss:.6f} best_val={best_val:.6f}"
            )

    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    if not best_path.is_file():
        _save_checkpoint(best_path, model, norm_x, norm_u, cfg, cfg.training.epochs, tr_loss)

    return best_path


def _run_epoch(
    model: DeepKoopman,
    loader: Optional[DataLoader],
    opt: Optional[torch.optim.Optimizer],
    device: torch.device,
    cfg: KoopmanConfig,
    train: bool,
) -> float:
    if loader is None:
        return float("nan")
    model.train(train)
    total = 0.0
    n = 0
    for x_seq, u_seq in loader:
        x_seq = x_seq.to(device)
        u_seq = u_seq.to(device)
        loss, _ = koopman_loss(
            model,
            x_seq,
            u_seq,
            one_step_weight=cfg.training.one_step_weight,
            multi_step_weight=cfg.training.multi_step_weight,
            recon_weight=cfg.training.recon_weight,
        )
        if train and opt is not None:
            opt.zero_grad()
            loss.backward()
            if cfg.training.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            opt.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


def _save_checkpoint(
    path: Path,
    model: DeepKoopman,
    norm_x: Normalizer,
    norm_u: Normalizer,
    cfg: KoopmanConfig,
    epoch: int,
    val_loss: float,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg.to_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "norm_x": {"mu": norm_x.mu.tolist(), "sigma": norm_x.sigma.tolist()},
            "norm_u": {"mu": norm_u.mu.tolist(), "sigma": norm_u.mu.tolist()},
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device | None = None) -> Tuple[DeepKoopman, KoopmanConfig, Normalizer, Normalizer, dict]:
    device = device or torch.device("cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg_dict = ckpt["config"]
    cfg = KoopmanConfig(
        Ts=cfg_dict["Ts"],
        nz=cfg_dict["nz"],
        nx=cfg_dict["nx"],
        nu=cfg_dict["nu"],
        horizon_N=cfg_dict["horizon_N"],
        ship_id=cfg_dict.get("ship_id", "bulkcarrier_65m"),
    )
    cfg.encoder.hidden = cfg_dict["encoder"]["hidden"]
    cfg.encoder.activation = cfg_dict["encoder"]["activation"]

    model = DeepKoopman(
        nx=cfg.nx,
        nu=cfg.nu,
        nz=cfg.nz,
        hidden=cfg.encoder.hidden,
        activation=cfg.encoder.activation,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    norm_x = Normalizer(
        mu=np.asarray(ckpt["norm_x"]["mu"]),
        sigma=np.asarray(ckpt["norm_x"]["sigma"]),
    )
    norm_u = Normalizer(
        mu=np.asarray(ckpt["norm_u"]["mu"]),
        sigma=np.asarray(ckpt["norm_u"]["sigma"]),
    )
    return model, cfg, norm_x, norm_u, ckpt
