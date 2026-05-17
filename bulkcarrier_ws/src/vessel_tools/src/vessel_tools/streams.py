"""Time-series containers extracted from ROS bags."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class SampleStream:
    """Irregular samples: times [N], values [N, D]."""

    times: np.ndarray
    values: np.ndarray
    name: str = ""

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times, dtype=np.float64).reshape(-1)
        self.values = np.asarray(self.values, dtype=np.float64)
        if self.values.ndim == 1:
            self.values = self.values.reshape(-1, 1)
        if len(self.times) != len(self.values):
            raise ValueError(f"{self.name}: times/values length mismatch")

    @property
    def dim(self) -> int:
        return int(self.values.shape[1])

    def sort_inplace(self) -> None:
        order = np.argsort(self.times)
        self.times = self.times[order]
        self.values = self.values[order]

    def dedupe_inplace(self) -> None:
        """Keep last sample when duplicate timestamps occur."""
        if len(self.times) < 2:
            return
        _, idx = np.unique(self.times, return_index=True)
        # unique returns first; we want last per timestamp
        rev_order = np.argsort(self.times)[::-1]
        rev_times = self.times[rev_order]
        rev_vals = self.values[rev_order]
        _, uidx = np.unique(rev_times, return_index=True)
        keep = rev_order[uidx]
        keep.sort()
        self.times = self.times[keep]
        self.values = self.values[keep]


@dataclass
class BagStreams:
    ins: SampleStream
    rudder_deg: SampleStream
    shaft_rpm: SampleStream
    wind: Optional[SampleStream] = None
    source_id: str = ""

    def all_streams(self) -> List[SampleStream]:
        out = [self.ins, self.rudder_deg, self.shaft_rpm]
        if self.wind is not None:
            out.append(self.wind)
        return out


@dataclass
class SyncedDataset:
    """Uniform grid aligned dataset."""

    t: np.ndarray
    x: np.ndarray  # [T, 6]  x,y,psi,u,v,r
    u: np.ndarray  # [T, 2]  delta_rad, rpm
    d: Optional[np.ndarray] = None  # [T, 2] wind speed, direction
    meta: Dict = field(default_factory=dict)
