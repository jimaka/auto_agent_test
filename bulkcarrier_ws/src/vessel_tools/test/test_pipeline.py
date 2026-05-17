#!/usr/bin/env python3
"""Integration tests: synthetic bag -> dataset."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PKG_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PKG_SRC))

from rosbags.rosbag1 import Writer
from vessel_tools.msgdefs import build_typestore
from vessel_tools.pipeline import bag_to_dataset, process_manifest
from vessel_tools.sync import sync_streams, chronological_splits
from vessel_tools.manifest import ProcessingSpec
from vessel_tools.streams import SampleStream, BagStreams


def _make_synthetic_bag(path: Path, duration: float = 10.0, dt: float = 0.1) -> None:
    store = build_typestore()
    t_arr = np.arange(0.0, duration, dt)
    path.parent.mkdir(parents=True, exist_ok=True)

    VesselState = store.types["vessel_msgs/msg/VesselState"]
    Float64 = store.types["std_msgs/msg/Float64"]
    Wind = store.types["vessel_msgs/msg/Wind"]
    Header = store.types["std_msgs/msg/Header"]
    BiTime = store.types["builtin_interfaces/msg/Time"]

    def header(stamp_ns: int):
        return Header(
            seq=0,
            stamp=BiTime(sec=stamp_ns // 1_000_000_000, nanosec=stamp_ns % 1_000_000_000),
            frame_id="local",
        )

    with Writer(path) as writer:
        c_ins = writer.add_connection("/ins/state", "vessel_msgs/msg/VesselState", typestore=store)
        c_rud = writer.add_connection("/sensors/rudder_deg", "std_msgs/msg/Float64", typestore=store)
        c_rpm = writer.add_connection("/sensors/shaft_rpm", "std_msgs/msg/Float64", typestore=store)
        c_wind = writer.add_connection("/sensors/wind", "vessel_msgs/msg/Wind", typestore=store)

        for i, t in enumerate(t_arr):
            stamp = int(t * 1e9)
            u = 5.0 + 0.1 * t
            msg_ins = VesselState(
                header=header(stamp),
                x=float(t * u * 0.1),
                y=0.0,
                psi=0.01 * t,
                u=u,
                v=0.1 * np.sin(t),
                r=0.01,
            )
            writer.write(c_ins, stamp, store.serialize_ros1(msg_ins, "vessel_msgs/msg/VesselState"))

            rud = Float64(data=5.0 * np.sin(0.2 * t))
            writer.write(c_rud, stamp, store.serialize_ros1(rud, "std_msgs/msg/Float64"))

            rpm = Float64(data=60.0 + 2.0 * t)
            writer.write(c_rpm, stamp, store.serialize_ros1(rpm, "std_msgs/msg/Float64"))

            w = Wind(header=header(stamp), speed_mps=3.0, direction_rad=0.5)
            writer.write(c_wind, stamp, store.serialize_ros1(w, "vessel_msgs/msg/Wind"))


class TestSync(unittest.TestCase):
    def test_chronological_splits(self):
        idx = chronological_splits(100, {"train": 0.7, "val": 0.15, "test": 0.15})
        self.assertEqual(len(idx["train"]), 70)
        self.assertEqual(len(idx["val"]), 15)
        self.assertEqual(len(idx["test"]), 15)


class TestPipeline(unittest.TestCase):
    def test_bag2dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            bag = Path(tmp) / "trial.bag"
            out = Path(tmp) / "processed"
            _make_synthetic_bag(bag)
            result = bag_to_dataset(bag, out, Ts=0.25, backend="rosbags")
            self.assertTrue((result / "train.npz").is_file())
            self.assertTrue((result / "meta.json").is_file())
            data = np.load(result / "train.npz")
            self.assertEqual(data["x"].shape[1], 6)
            self.assertEqual(data["u"].shape[1], 2)
            # Rudder converted to rad
            self.assertTrue(np.max(np.abs(data["u"][:, 0])) < 1.0)

    def test_manifest_pipeline(self):
        repo = Path(__file__).resolve().parents[4]
        with tempfile.TemporaryDirectory() as tmp:
            bag = Path(tmp) / "example.bag"
            _make_synthetic_bag(bag, duration=20.0)

            manifest = Path(tmp) / "manifest.yaml"
            processed = Path(tmp) / "processed"
            manifest.write_text(
                f"""
dataset_id: test_synthetic
ship_id: bulkcarrier_65m
Ts: 0.25
sources:
  - type: rosbag
    path: {bag}
topics:
  ins: /ins/state
  rudder_deg: /sensors/rudder_deg
  shaft_rpm: /sensors/shaft_rpm
  wind: /sensors/wind
output:
  root: {processed}
  format: npz
splits:
  train: 0.7
  val: 0.15
  test: 0.15
""",
                encoding="utf-8",
            )
            out = process_manifest(manifest, repo_root=tmp, backend="rosbags")
            self.assertTrue((out / "train.npz").is_file())
            self.assertTrue((out / "val.npz").is_file())
            self.assertTrue((out / "test.npz").is_file())


if __name__ == "__main__":
    unittest.main()
