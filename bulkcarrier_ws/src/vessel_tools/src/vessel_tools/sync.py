"""Resample and align streams to a uniform grid."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from vessel_tools.manifest import ProcessingSpec
from vessel_tools.streams import BagStreams, SampleStream, SyncedDataset

DEG2RAD = math.pi / 180.0


def _interp_series(stream: SampleStream, grid: np.ndarray, max_gap: float) -> np.ndarray:
    if len(stream.times) == 0:
        raise ValueError(f"Stream {stream.name} is empty")
    out = np.zeros((len(grid), stream.dim), dtype=np.float64)
    for d in range(stream.dim):
        out[:, d] = np.interp(grid, stream.times, stream.values[:, d])
    _warn_gaps(stream, grid, max_gap)
    return out


def _warn_gaps(stream: SampleStream, grid: np.ndarray, max_gap: float) -> None:
    if len(stream.times) < 2:
        return
    dt = np.diff(stream.times)
    if np.max(dt) > max_gap:
        import warnings

        warnings.warn(
            f"Stream '{stream.name}' has gap {np.max(dt):.3f}s > max_gap {max_gap}s",
            RuntimeWarning,
            stacklevel=2,
        )


def common_time_range(streams: List[SampleStream]) -> Tuple[float, float]:
    t0 = max(float(s.times[0]) for s in streams if len(s.times))
    t1 = min(float(s.times[-1]) for s in streams if len(s.times))
    if t1 <= t0:
        raise ValueError(f"Invalid overlap: t0={t0}, t1={t1}")
    return t0, t1


def make_grid(t0: float, t1: float, Ts: float) -> np.ndarray:
    n = int(np.floor((t1 - t0) / Ts)) + 1
    if n < 2:
        raise ValueError(f"Grid too short: duration={t1 - t0}, Ts={Ts}")
    grid = t0 + np.arange(n, dtype=np.float64) * Ts
    if grid[-1] > t1 + 1e-9:
        grid = grid[grid <= t1 + 1e-9]
    return grid


def sync_streams(
    streams: BagStreams,
    Ts: float,
    processing: ProcessingSpec,
) -> SyncedDataset:
    for st in streams.all_streams():
        st.sort_inplace()
        st.dedupe_inplace()

    active = [streams.ins, streams.rudder_deg, streams.shaft_rpm]
    if streams.wind is not None:
        active.append(streams.wind)

    t0, t1 = common_time_range(active)
    grid = make_grid(t0, t1, Ts)
    max_gap = processing.max_gap_sec

    x = _interp_series(streams.ins, grid, max_gap)
    rud_deg = _interp_series(streams.rudder_deg, grid, max_gap)
    rpm = _interp_series(streams.shaft_rpm, grid, max_gap)

    delta_rad = rud_deg[:, 0] * DEG2RAD if processing.rudder_in_degrees else rud_deg[:, 0]
    u = np.column_stack([delta_rad, rpm[:, 0]])

    d = None
    if streams.wind is not None:
        d = _interp_series(streams.wind, grid, max_gap)

    # Wrap heading to [-pi, pi] after interpolation
    x[:, 2] = np.arctan2(np.sin(x[:, 2]), np.cos(x[:, 2]))

    return SyncedDataset(
        t=grid,
        x=x,
        u=u,
        d=d,
        meta={
            "t0": t0,
            "t1": t1,
            "Ts": Ts,
            "n_samples": len(grid),
            "source_id": streams.source_id,
        },
    )


def chronological_splits(
    n: int, splits: dict[str, float]
) -> dict[str, np.ndarray]:
    """Return index arrays for train/val/test along time."""
    order = ["train", "val", "test"]
    keys = [k for k in order if k in splits]
    fracs = [splits[k] for k in keys]
    bounds = np.floor(np.cumsum([0.0] + fracs) * n).astype(int)
    out = {}
    for i, key in enumerate(keys):
        start, end = bounds[i], bounds[i + 1]
        if i == len(keys) - 1:
            end = n
        out[key] = np.arange(start, end, dtype=np.int64)
    return out
