"""Export deployment bundle for vessel_control."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml

from vessel_identification.config import KoopmanConfig
from vessel_identification.models.koopman import DeepKoopman, MlpEncoder
from vessel_identification.normalizer import Normalizer
from vessel_identification.trainer import load_checkpoint
from vessel_identification.dataset import load_npz_split
from vessel_identification.validate import horizon_metrics, write_tube_csv


class EncoderOnnxWrapper(torch.nn.Module):
    """ONNX: normalized x [B, nx] -> z [B, nz]."""

    def __init__(self, encoder: MlpEncoder) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        return self.encoder(x_norm)


def _write_f64_bin(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(arr, dtype=np.float64)
    arr.tofile(path)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_bundle(
    checkpoint: Path,
    registry: Path,
    model_id: str,
    data_dir: Path | None = None,
    description: str = "",
    opset: int = 18,
) -> Path:
    device = torch.device("cpu")
    model, cfg, norm_x, norm_u, ckpt = load_checkpoint(checkpoint, device)
    out_dir = registry / model_id
    out_dir.mkdir(parents=True, exist_ok=True)

    norm_x.save(out_dir / "norm_x.json")
    norm_u.save(out_dir / "norm_u.json")

    a = model.A.detach().cpu().numpy().astype(np.float64)
    b = model.B.detach().cpu().numpy().astype(np.float64)
    _write_f64_bin(out_dir / "A.bin", a)
    _write_f64_bin(out_dir / "B.bin", b)

    # Decoder weights as Cx: x ~= z @ Cx.T  => Cx shape [nx, nz]
    dec_w = model.decoder.weight.detach().cpu().numpy().astype(np.float64)
    dec_b = model.decoder.bias.detach().cpu().numpy().astype(np.float64)
    _write_f64_bin(out_dir / "Cx.bin", dec_w)
    with (out_dir / "decoder_bias.json").open("w", encoding="utf-8") as f:
        json.dump({"bias": dec_b.tolist()}, f)

    wrapper = EncoderOnnxWrapper(model.encoder).eval()
    dummy = torch.zeros(1, cfg.nx, dtype=torch.float32)
    onnx_path = out_dir / "encoder.onnx"
    export_kw = dict(
        input_names=["x_norm"],
        output_names=["z"],
        dynamic_axes={"x_norm": {0: "batch"}, "z": {0: "batch"}},
        opset_version=opset,
    )
    try:
        torch.onnx.export(wrapper, dummy, str(onnx_path), dynamo=False, **export_kw)
    except TypeError:
        torch.onnx.export(wrapper, dummy, str(onnx_path), **export_kw)

    val_metrics: Dict = {}
    if data_dir is not None and (data_dir / "val.npz").is_file():
        arrays = load_npz_split(data_dir, "val")
        hm = horizon_metrics(
            model, norm_x, norm_u, arrays["x"], arrays["u"], cfg.horizon_N, device
        )
        val_metrics = {
            "n_step_z_rmse": hm["n_step_z_rmse"],
            "n_step_x_rmse": hm["n_step_x_rmse"],
        }
        write_tube_csv(out_dir / "tube_tightening.csv", hm["x_rmse_per_step"])

    meta = {
        "schema_version": "1.0",
        "model_id": model_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ship_id": cfg.ship_id,
        "description": description or f"Exported from {checkpoint.name}",
        "Ts": cfg.Ts,
        "nz": cfg.nz,
        "nx": cfg.nx,
        "nu": cfg.nu,
        "nd": 0,
        "horizon_N": cfg.horizon_N,
        "state_names": ["x", "y", "psi", "u", "v", "r"],
        "state_units": ["m", "m", "rad", "m/s", "m/s", "rad/s"],
        "input_names": ["delta", "n"],
        "input_units": ["rad", "rpm"],
        "normalization": {
            "x": {"file": "norm_x.json", "method": "standard"},
            "u": {"file": "norm_u.json", "method": "standard"},
            "z": {"method": "none"},
        },
        "files": {
            "A": {"path": "A.bin", "dtype": "float64", "shape": [cfg.nz, cfg.nz]},
            "B": {"path": "B.bin", "dtype": "float64", "shape": [cfg.nz, cfg.nu]},
            "encoder": {
                "path": "encoder.onnx",
                "opset": opset,
                "input": "x_norm",
                "output": "z",
            },
            "Cx": {"path": "Cx.bin", "dtype": "float64", "shape": [cfg.nx, cfg.nz], "optional": True},
            "tube": {"path": "tube_tightening.csv"},
        },
        "validation": {
            "checkpoint_epoch": int(ckpt.get("epoch", 0)),
            "checkpoint_val_loss": float(ckpt.get("val_loss", float("nan"))),
            "sil_passed": False,
            **val_metrics,
        },
        "baseline_fallback": "nomoto_mpc_v1",
    }

    with (out_dir / "meta.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False)

    checksums = {}
    for p in sorted(out_dir.iterdir()):
        if p.is_file() and p.name != "checksums.sha256":
            checksums[p.name] = _sha256_file(p)
    with (out_dir / "checksums.sha256").open("w", encoding="utf-8") as f:
        for name, digest in checksums.items():
            f.write(f"{digest}  {name}\n")

    return out_dir
