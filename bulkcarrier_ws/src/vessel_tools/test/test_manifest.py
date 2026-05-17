#!/usr/bin/env python3
import os
import sys
import unittest
from pathlib import Path

PKG_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PKG_SRC))

from vessel_tools.manifest import load_manifest  # noqa: E402


class TestManifest(unittest.TestCase):
    def test_load_example(self):
        repo = Path(__file__).resolve().parents[4]
        manifest_path = repo / "data" / "manifests" / "example_trial.yaml"
        if not manifest_path.is_file():
            self.skipTest("example manifest missing")
        m = load_manifest(manifest_path, repo_root=repo)
        self.assertEqual(m.dataset_id, "ship_trials_example")
        self.assertEqual(m.Ts, 0.25)
        self.assertAlmostEqual(sum(m.splits.values()), 1.0)
        self.assertEqual(m.output.format, "npz")


if __name__ == "__main__":
    unittest.main()
