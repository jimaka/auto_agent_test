#!/usr/bin/env python3
"""SIL benchmark smoke tests."""
import sys
import unittest
from pathlib import Path

PKG_SRC = Path(__file__).resolve().parents[1] / "src"
WS = PKG_SRC.parents[2]  # bulkcarrier_ws
sys.path.insert(0, str(PKG_SRC))

from vessel_simulation.mmg3dof import MMG3DOF, VesselState, ControlInput
from vessel_simulation.metrics import cross_track_error
from vessel_simulation.sil_runner import load_sil_config, run_sil
from vessel_simulation.trajectory import StraightTrajectory
from vessel_simulation.controllers.nomoto import NomotoController


class TestMMG(unittest.TestCase):
    def test_plant_step(self):
        cfg = WS / "src" / "vessel_simulation" / "config" / "mmg_default.yaml"
        if not cfg.is_file():
            self.skipTest("config missing")
        plant = MMG3DOF.from_yaml(str(cfg))
        s0 = VesselState(0, 0, 0, 5, 0, 0)
        s1 = plant.step(s0, ControlInput(0.1, 70))
        self.assertGreater(s1.x, s0.x)


class TestSIL(unittest.TestCase):
    def test_nomoto_straight_short(self):
        cfg_path = WS / "src" / "vessel_simulation" / "config" / "mmg_default.yaml"
        if not cfg_path.is_file():
            self.skipTest("config missing")

        sil_cfg = load_sil_config(cfg_path)
        plant = MMG3DOF.from_yaml(str(cfg_path))
        traj = StraightTrajectory(u_ref=sil_cfg.u0, duration=30.0)
        ctrl = NomotoController()
        res = run_sil(plant, traj, ctrl, sil_cfg, "nomoto", duration_s=30.0)
        self.assertGreater(res.metrics.n_samples, 10)
        self.assertTrue(res.metrics.e_cross_rms < 50.0)

    def test_koopman_short(self):
        cfg_path = WS / "src" / "vessel_simulation" / "config" / "mmg_default.yaml"
        model_dir = WS / "src" / "vessel_control" / "model_registry" / "koopman_test"
        if not cfg_path.is_file() or not model_dir.is_dir():
            self.skipTest("assets missing")
        try:
            from vessel_simulation.controllers.koopman_mpc import KoopmanMpcController
        except ImportError:
            self.skipTest("onnxruntime/osqp not installed")

        sil_cfg = load_sil_config(cfg_path)
        plant = MMG3DOF.from_yaml(str(cfg_path))
        traj = StraightTrajectory(u_ref=sil_cfg.u0, duration=40.0)
        ctrl = KoopmanMpcController(model_dir)
        res = run_sil(plant, traj, ctrl, sil_cfg, "koopman_mpc", duration_s=40.0)
        self.assertGreater(res.metrics.n_samples, 10)


if __name__ == "__main__":
    unittest.main()
