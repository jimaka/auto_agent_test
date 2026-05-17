"""Load processed NPZ datasets for Koopman training."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from vessel_identification.normalizer import Normalizer


def load_npz_split(data_dir: Path, split: str) -> Dict[str, np.ndarray]:
    path = data_dir / f"{split}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing split file: {path}")
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def load_all_splits(data_dir: Path) -> Dict[str, Dict[str, np.ndarray]]:
    out = {}
    for split in ("train", "val", "test"):
        p = data_dir / f"{split}.npz"
        if p.is_file():
            out[split] = load_npz_split(data_dir, split)
    if "train" not in out:
        raise FileNotFoundError(f"No train.npz under {data_dir}")
    return out


def concatenate_series(splits: Dict[str, Dict[str, np.ndarray]], keys: List[str]) -> Dict[str, np.ndarray]:
    arrays = {k: [] for k in keys}
    for split in sorted(splits.keys()):
        for k in keys:
            arrays[k].append(splits[split][k])
    return {k: np.concatenate(v, axis=0) for k, v in arrays.items()}


class KoopmanSequenceDataset(Dataset):
    """Sliding windows: x[T, nx], u[T, nu] with horizon N."""

    def __init__(
        self,
        x: np.ndarray,
        u: np.ndarray,
        horizon: int,
        norm_x: Normalizer,
        norm_u: Normalizer,
    ) -> None:
        self.x = x.astype(np.float32)
        self.u = u.astype(np.float32)
        self.horizon = horizon
        self.norm_x = norm_x
        self.norm_u = norm_u
        self.n_windows = len(x) - horizon
        if self.n_windows < 1:
            raise ValueError(f"Series too short ({len(x)}) for horizon {horizon}")

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sl = slice(idx, idx + self.horizon + 1)
        x_win = self.norm_x.normalize(self.x[sl])
        u_win = self.norm_u.normalize(self.u[idx : idx + self.horizon])
        return (
            torch.from_numpy(x_win.astype(np.float32)),
            torch.from_numpy(u_win.astype(np.float32)),
        )


def build_dataloaders(
    data_dir: Path,
    horizon: int,
    batch_size: int,
    num_workers: int = 0,
) -> Tuple[Dict[str, KoopmanSequenceDataset], Normalizer, Normalizer]:
    splits = load_all_splits(data_dir)
    train = splits["train"]
    norm_x = Normalizer.fit(train["x"])
    norm_u = Normalizer.fit(train["u"])

    datasets: Dict[str, KoopmanSequenceDataset] = {}
    for name, arrays in splits.items():
        datasets[name] = KoopmanSequenceDataset(
            arrays["x"], arrays["u"], horizon, norm_x, norm_u
        )

    return datasets, norm_x, norm_u
