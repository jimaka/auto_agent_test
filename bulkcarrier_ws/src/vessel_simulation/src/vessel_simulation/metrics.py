"""Tracking performance metrics for SIL benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np

from vessel_simulation.mmg3dof import VesselState, wrap_pi


@dataclass
class TrackingSample:
    t: float
    e_cross: float
    e_psi: float
    e_u: float


@dataclass
class BenchmarkMetrics:
    e_cross_rms: float
    e_psi_rms: float
    e_u_rms: float
    max_overshoot_cross: float
    max_overshoot_u: float
    duration_s: float
    n_samples: int

    def to_dict(self) -> Dict:
        d = asdict(self)
        return {k: float(v) if isinstance(v, (np.floating, float)) else int(v) for k, v in d.items()}


def cross_track_error(state: VesselState, ref: VesselState) -> float:
    dx = state.x - ref.x
    dy = state.y - ref.y
    return float(-np.sin(ref.psi) * dx + np.cos(ref.psi) * dy)


def heading_error(state: VesselState, ref: VesselState) -> float:
    return wrap_pi(state.psi - ref.psi)


def speed_error(state: VesselState, ref: VesselState) -> float:
    return state.u - ref.u


def compute_sample(state: VesselState, ref: VesselState, t: float) -> TrackingSample:
    return TrackingSample(
        t=t,
        e_cross=cross_track_error(state, ref),
        e_psi=heading_error(state, ref),
        e_u=speed_error(state, ref),
    )


def aggregate_metrics(samples: List[TrackingSample]) -> BenchmarkMetrics:
    if not samples:
        return BenchmarkMetrics(0, 0, 0, 0, 0, 0, 0)
    ec = np.array([s.e_cross for s in samples])
    ep = np.array([s.e_psi for s in samples])
    eu = np.array([s.e_u for s in samples])
    return BenchmarkMetrics(
        e_cross_rms=float(np.sqrt(np.mean(ec**2))),
        e_psi_rms=float(np.sqrt(np.mean(ep**2))),
        e_u_rms=float(np.sqrt(np.mean(eu**2))),
        max_overshoot_cross=float(np.max(np.abs(ec))),
        max_overshoot_u=float(np.max(np.abs(eu))),
        duration_s=float(samples[-1].t - samples[0].t),
        n_samples=len(samples),
    )
