#!/usr/bin/env python3
"""Smoke tests: train, validate, export on synthetic NPZ data."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PKG_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PKG_SRC))

from vessel_identification.config import KoopmanConfig  # noqa: E402
from vessel_identification.export import export_bundle  # noqa: E402
from vessel_identification.trainer import train_model  # noqa: E402
from vessel_identification.validate import validate_checkpoint  # noqa: E402


def _synthetic_dataset(path: Path, n: int = 800) -> None:
    t = np.arange(n, dtype=np.float64) * 0.25
    u = np.column_stack([0.1 * np.sin(0.1 * t), 60.0 + 0.05 * t])
    x = np.zeros((n, 6))
    x[:, 0] = np.cumsum(5.0 * np.cos(0.05 * t) * 0.25)
    x[:, 1] = np.cumsum(5.0 * np.sin(0.05 * t) * 0.25)
    x[:, 2] = 0.05 * t
    x[:, 3] = 5.0 + 0.1 * np.sin(0.05 * t)
    x[:, 4] = 0.1 * np.cos(0.1 * t)
    x[:, 5] = 0.01
    path.mkdir(parents=True, exist_ok=True)
    n_tr = int(0.7 * n)
    n_va = int(0.15 * n)
    np.savez_compressed(path / "train.npz", t=t[:n_tr], x=x[:n_tr], u=u[:n_tr])
    np.savez_compressed(
        path / "val.npz",
        t=t[n_tr : n_tr + n_va],
        x=x[n_tr : n_tr + n_va],
        u=u[n_tr : n_tr + n_va],
    )
    np.savez_compressed(path / "test.npz", t=t[n_tr + n_va :], x=x[n_tr + n_va :], u=u[n_tr + n_va :])


class TestKoopmanPipeline(unittest.TestCase):
    def test_train_validate_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "data"
            run_dir = tmp_path / "runs"
            registry = tmp_path / "registry"
            _synthetic_dataset(data_dir)

            from vessel_identification.config import EncoderConfig, TrainingConfig

            cfg = KoopmanConfig(
                nz=16,
                horizon_N=10,
                encoder=EncoderConfig(hidden=[32, 32]),
                training=TrainingConfig(epochs=5, batch_size=64),
            )

            ckpt = train_model(cfg, data_dir, run_dir)
            self.assertTrue(ckpt.is_file())

            summary, _, _, _ = validate_checkpoint(ckpt, data_dir, split="val")
            self.assertIn("n_step_z_rmse", summary)

            out = export_bundle(
                ckpt,
                registry,
                "koopman_test_001",
                data_dir=data_dir,
            )
            self.assertTrue((out / "encoder.onnx").is_file())
            self.assertTrue((out / "encoder_io.json").is_file())
            self.assertTrue((out / "A.bin").is_file())
            self.assertTrue((out / "meta.yaml").is_file())
            self.assertTrue((out / "checksums.sha256").is_file())

            import json
            import onnxruntime as ort

            with (out / "encoder_io.json").open("r", encoding="utf-8") as f:
                io_doc = json.load(f)
            sess = ort.InferenceSession(str(out / "encoder.onnx"))
            inp = np.zeros((1, 6), dtype=np.float32)
            z = sess.run(None, {io_doc["input_name"]: inp})[0]
            self.assertEqual(z.shape, (1, 16))


if __name__ == "__main__":
    unittest.main()
