#!/usr/bin/env python3
"""Tests for PyTorch -> ONNX conversion and encoder_io.json contract."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PKG_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PKG_SRC))

from vessel_identification.config import EncoderConfig, KoopmanConfig, TrainingConfig  # noqa: E402
from vessel_identification.onnx_convert import (  # noqa: E402
    convert_checkpoint_to_onnx,
    export_encoder_onnx,
    write_encoder_io_manifest,
)
from vessel_identification.trainer import train_model  # noqa: E402
from test_koopman import _synthetic_dataset  # noqa: E402


class TestOnnxConvert(unittest.TestCase):
    def test_export_encoder_onnx_writes_io_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "data"
            run_dir = tmp_path / "runs"
            _synthetic_dataset(data_dir, n=400)

            cfg = KoopmanConfig(
                nz=8,
                horizon_N=5,
                encoder=EncoderConfig(hidden=[16, 16]),
                training=TrainingConfig(epochs=3, batch_size=32),
            )
            ckpt = train_model(cfg, data_dir, run_dir)
            out_dir = tmp_path / "onnx_only"
            result = convert_checkpoint_to_onnx(ckpt, out_dir)
            self.assertTrue(result.onnx_path.is_file())
            self.assertTrue(result.io_path.is_file())

            import json
            import onnxruntime as ort

            with result.io_path.open("r", encoding="utf-8") as f:
                io_doc = json.load(f)
            self.assertEqual(io_doc["input_name"], "x_norm")
            self.assertEqual(io_doc["output_name"], "z")
            self.assertEqual(io_doc["nx"], 6)
            self.assertEqual(io_doc["nz"], 8)

            sess = ort.InferenceSession(str(result.onnx_path))
            x = np.random.randn(1, 6).astype(np.float32)
            z = sess.run(None, {io_doc["input_name"]: x})[0]
            self.assertEqual(z.shape, (1, 8))

    def test_write_encoder_io_manifest_from_existing_onnx(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "data"
            run_dir = tmp_path / "runs"
            _synthetic_dataset(data_dir, n=200)
            cfg = KoopmanConfig(
                nz=4,
                horizon_N=3,
                encoder=EncoderConfig(hidden=[8]),
                training=TrainingConfig(epochs=2, batch_size=16),
            )
            ckpt = train_model(cfg, data_dir, run_dir)
            from vessel_identification.trainer import load_checkpoint
            import torch

            model, cfg_loaded, _, _, _ = load_checkpoint(ckpt, torch.device("cpu"))
            onnx_dir = tmp_path / "bundle"
            export_encoder_onnx(model, cfg_loaded, onnx_dir, verify=False)
            io2 = tmp_path / "encoder_io_copy.json"
            write_encoder_io_manifest(
                onnx_dir / "encoder.onnx", io2, nx=cfg_loaded.nx, nz=cfg_loaded.nz
            )
            self.assertTrue(io2.is_file())


if __name__ == "__main__":
    unittest.main()
