"""Load Koopman training configuration from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class EncoderConfig:
    hidden: List[int] = field(default_factory=lambda: [128, 128])
    activation: str = "relu"


@dataclass
class TrainingConfig:
    batch_size: int = 256
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-5
    multi_step_weight: float = 1.0
    one_step_weight: float = 1.0
    recon_weight: float = 0.5
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "auto"


@dataclass
class KoopmanConfig:
    Ts: float = 0.25
    nz: int = 64
    nx: int = 6
    nu: int = 2
    horizon_N: int = 30
    ship_id: str = "bulkcarrier_65m"
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "KoopmanConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            raw: Dict[str, Any] = yaml.safe_load(f)
        enc = raw.get("encoder", {})
        tr = raw.get("training", {})
        return cls(
            Ts=float(raw.get("Ts", 0.25)),
            nz=int(raw.get("nz", 64)),
            nx=int(raw.get("nx", 6)),
            nu=int(raw.get("nu", 2)),
            horizon_N=int(raw.get("horizon_N", 30)),
            ship_id=str(raw.get("ship_id", "bulkcarrier_65m")),
            encoder=EncoderConfig(
                hidden=list(enc.get("hidden", [128, 128])),
                activation=str(enc.get("activation", "relu")),
            ),
            training=TrainingConfig(
                batch_size=int(tr.get("batch_size", 256)),
                epochs=int(tr.get("epochs", 200)),
                lr=float(tr.get("lr", 1e-3)),
                weight_decay=float(tr.get("weight_decay", 1e-5)),
                multi_step_weight=float(tr.get("multi_step_weight", 1.0)),
                one_step_weight=float(tr.get("one_step_weight", 1.0)),
                recon_weight=float(tr.get("recon_weight", 0.5)),
                grad_clip=float(tr.get("grad_clip", 1.0)),
                seed=int(tr.get("seed", 42)),
                device=str(tr.get("device", "auto")),
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Ts": self.Ts,
            "nz": self.nz,
            "nx": self.nx,
            "nu": self.nu,
            "horizon_N": self.horizon_N,
            "ship_id": self.ship_id,
            "encoder": {"hidden": self.encoder.hidden, "activation": self.encoder.activation},
            "training": {
                "batch_size": self.training.batch_size,
                "epochs": self.training.epochs,
                "lr": self.training.lr,
                "weight_decay": self.training.weight_decay,
                "multi_step_weight": self.training.multi_step_weight,
                "one_step_weight": self.training.one_step_weight,
                "recon_weight": self.training.recon_weight,
                "grad_clip": self.training.grad_clip,
                "seed": self.training.seed,
                "device": self.training.device,
            },
        }
