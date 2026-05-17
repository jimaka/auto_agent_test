"""PyTorch encoder -> ONNX conversion with ORT parity verification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from vessel_identification.config import KoopmanConfig
from vessel_identification.models.koopman import DeepKoopman, MlpEncoder
from vessel_identification.normalizer import Normalizer
from vessel_identification.trainer import load_checkpoint

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    onnx = None
    ort = None


class EncoderOnnxWrapper(torch.nn.Module):
    """Deployment graph: x_norm [B, nx] -> z [B, nz]."""

    def __init__(self, encoder: MlpEncoder) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        return self.encoder(x_norm)


@dataclass
class OnnxConvertResult:
    onnx_path: Path
    io_path: Path
    max_abs_err: float
    opset: int
    input_name: str
    output_name: str
    nz: int
    nx: int


def export_encoder_onnx(
    model: DeepKoopman,
    cfg: KoopmanConfig,
    out_dir: Path,
    onnx_filename: str = "encoder.onnx",
    io_filename: str = "encoder_io.json",
    opset: int = 18,
    atol: float = 1e-5,
    verify: bool = True,
) -> OnnxConvertResult:
    """Export encoder to ONNX and write encoder_io.json for C++ ORT loader."""
    if onnx is None or ort is None:
        raise ImportError("onnx and onnxruntime are required for ONNX conversion")

    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / onnx_filename
    io_path = out_dir / io_filename

    wrapper = EncoderOnnxWrapper(model.encoder).eval()
    dummy = torch.zeros(1, cfg.nx, dtype=torch.float32)
    input_name, output_name = "x_norm", "z"

    export_kw = dict(
        input_names=[input_name],
        output_names=[output_name],
        dynamic_axes={input_name: {0: "batch"}, output_name: {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
    )
    try:
        torch.onnx.export(wrapper, dummy, str(onnx_path), dynamo=False, **export_kw)
    except TypeError:
        torch.onnx.export(wrapper, dummy, str(onnx_path), **export_kw)

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    max_err = 0.0
    if verify:
        max_err = _verify_onnx_parity(wrapper, onnx_path, cfg.nx, cfg.nz)

    io_doc: Dict[str, Any] = {
        "format_version": "1.0",
        "opset": opset,
        "input_name": input_name,
        "output_name": output_name,
        "nx": cfg.nx,
        "nz": cfg.nz,
        "dtype": "float32",
        "input_shape": [1, cfg.nx],
        "output_shape": [1, cfg.nz],
        "dynamic_batch": True,
        "max_abs_err_torch_vs_ort": max_err,
    }
    with io_path.open("w", encoding="utf-8") as f:
        json.dump(io_doc, f, indent=2)

    if max_err > atol:
        raise RuntimeError(
            f"ONNX parity check failed: max_abs_err={max_err:.3e} > atol={atol:.3e}"
        )

    return OnnxConvertResult(
        onnx_path=onnx_path,
        io_path=io_path,
        max_abs_err=max_err,
        opset=opset,
        input_name=input_name,
        output_name=output_name,
        nz=cfg.nz,
        nx=cfg.nx,
    )


def convert_checkpoint_to_onnx(
    checkpoint: Path,
    out_dir: Path,
    opset: int = 18,
    atol: float = 1e-5,
) -> OnnxConvertResult:
    """Load PyTorch checkpoint and export verified ONNX encoder."""
    model, cfg, _, _, _ = load_checkpoint(checkpoint, torch.device("cpu"))
    return export_encoder_onnx(
        model, cfg, out_dir, opset=opset, atol=atol, verify=True
    )


def write_encoder_io_manifest(
    onnx_path: Path,
    io_path: Path,
    *,
    nx: int,
    nz: int,
    opset: int = 18,
    input_name: str = "x_norm",
    output_name: str = "z",
) -> Path:
    """Write encoder_io.json for an existing ONNX file (no PyTorch required)."""
    if ort is None:
        raise ImportError("onnxruntime is required")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    io_doc: Dict[str, Any] = {
        "format_version": "1.0",
        "opset": opset,
        "input_name": inp.name or input_name,
        "output_name": out.name or output_name,
        "nx": nx,
        "nz": nz,
        "dtype": "float32",
        "input_shape": list(inp.shape),
        "output_shape": list(out.shape),
        "dynamic_batch": bool(inp.shape) and "batch" in str(inp.shape[0]).lower(),
    }
    io_path.parent.mkdir(parents=True, exist_ok=True)
    with io_path.open("w", encoding="utf-8") as f:
        json.dump(io_doc, f, indent=2)
    return io_path


def _verify_onnx_parity(
    wrapper: EncoderOnnxWrapper,
    onnx_path: Path,
    nx: int,
    nz: int,
    n_samples: int = 32,
) -> float:
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    max_err = 0.0
    with torch.no_grad():
        for _ in range(n_samples):
            x = rng.standard_normal(nx).astype(np.float32)
            z_torch = wrapper(torch.from_numpy(x.reshape(1, -1))).numpy().reshape(-1)[:nz]
            z_ort = sess.run(None, {"x_norm": x.reshape(1, -1)})[0].reshape(-1)[:nz]
            max_err = max(max_err, float(np.max(np.abs(z_torch - z_ort))))
    return max_err
