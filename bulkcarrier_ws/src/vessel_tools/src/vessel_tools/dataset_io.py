"""Write processed datasets to disk."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from vessel_tools.manifest import Manifest
from vessel_tools.streams import SyncedDataset


def write_split_npz(path: Path, data: SyncedDataset, indices: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "t": data.t[indices],
        "x": data.x[indices],
        "u": data.u[indices],
    }
    if data.d is not None:
        payload["d"] = data.d[indices]
    np.savez_compressed(path, **payload)


def write_split_csv(path: Path, data: SyncedDataset, indices: np.ndarray) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["t", "x", "y", "psi", "u", "v", "r", "delta_rad", "rpm"]
    if data.d is not None:
        cols.extend(["wind_speed", "wind_dir"])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in indices:
            row = [
                data.t[i],
                *data.x[i].tolist(),
                data.u[i, 0],
                data.u[i, 1],
            ]
            if data.d is not None:
                row.extend(data.d[i].tolist())
            w.writerow(row)


def export_dataset(
    data: SyncedDataset,
    manifest: Manifest,
    split_indices: Dict[str, np.ndarray],
) -> Path:
    out_root = manifest.output.root
    out_root.mkdir(parents=True, exist_ok=True)

    fmt = manifest.output.format
    for split, idx in split_indices.items():
        if len(idx) == 0:
            continue
        if fmt == "npz":
            write_split_npz(out_root / f"{split}.npz", data, idx)
        elif fmt == "csv":
            write_split_csv(out_root / f"{split}.csv", data, idx)
        else:
            raise ValueError(f"Unsupported output format: {fmt}")

    meta = {
        "dataset_id": manifest.dataset_id,
        "ship_id": manifest.ship_id,
        "Ts": manifest.Ts,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "state_names": ["x", "y", "psi", "u", "v", "r"],
        "input_names": ["delta_rad", "rpm"],
        "n_samples": int(len(data.t)),
        "splits": {k: int(len(v)) for k, v in split_indices.items()},
        "stream_meta": data.meta,
        "topics": {
            "ins": manifest.topics.ins,
            "rudder_deg": manifest.topics.rudder_deg,
            "shaft_rpm": manifest.topics.shaft_rpm,
            "wind": manifest.topics.wind,
        },
    }
    if data.d is not None:
        meta["disturbance_names"] = ["wind_speed_mps", "wind_direction_rad"]

    with (out_root / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if manifest.manifest_path:
        shutil.copy2(manifest.manifest_path, out_root / "manifest.yaml")

    return out_root
