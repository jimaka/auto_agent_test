"""Time-parameterized reference trajectories for SIL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from vessel_simulation.mmg3dof import VesselState, wrap_pi


@dataclass
class TrajectoryRef:
    """Callable x_ref(t) -> VesselState."""

    t0: float = 0.0
    duration: float = 600.0

    def state_at(self, t: float) -> VesselState:
        raise NotImplementedError


class StraightTrajectory(TrajectoryRef):
    def __init__(self, u_ref: float = 5.0, psi_ref: float = 0.0, duration: float = 600.0) -> None:
        self.u_ref = u_ref
        self.psi_ref = psi_ref
        self.duration = duration

    def state_at(self, t: float) -> VesselState:
        return VesselState(
            x=self.u_ref * np.cos(self.psi_ref) * t,
            y=self.u_ref * np.sin(self.psi_ref) * t,
            psi=self.psi_ref,
            u=self.u_ref,
            v=0.0,
            r=0.0,
        )


class TurnTrajectory(TrajectoryRef):
    def __init__(self, u_ref: float = 5.0, turn_rate: float = 0.02, duration: float = 600.0) -> None:
        self.u_ref = u_ref
        self.turn_rate = turn_rate
        self.duration = duration

    def state_at(self, t: float) -> VesselState:
        psi = wrap_pi(self.turn_rate * t)
        if abs(self.turn_rate) < 1e-6:
            return StraightTrajectory(self.u_ref, 0.0).state_at(t)
        R = self.u_ref / self.turn_rate
        x = R * np.sin(psi)
        y = R * (1.0 - np.cos(psi))
        return VesselState(x=x, y=y, psi=psi, u=self.u_ref, v=0.0, r=self.turn_rate)


class ZigzagTrajectory(TrajectoryRef):
    """Piecewise heading changes (10-10 zigzag pattern)."""

    def __init__(self, u_ref: float = 5.0, amplitude_deg: float = 15.0, leg_s: float = 80.0,
                 duration: float = 600.0) -> None:
        self.u_ref = u_ref
        self.amp = np.deg2rad(amplitude_deg)
        self.leg_s = leg_s
        self.duration = duration

    def state_at(self, t: float) -> VesselState:
        leg = int(t / self.leg_s)
        sign = 1.0 if leg % 2 == 0 else -1.0
        psi = sign * self.amp
        x = self.u_ref * np.cos(psi) * t
        y = self.u_ref * np.sin(psi) * t
        return VesselState(x=x, y=y, psi=psi, u=self.u_ref, v=0.0, r=0.0)


def make_scenario(name: str, u_ref: float = 5.0, duration: float = 600.0) -> TrajectoryRef:
    name = name.lower()
    if name in ("straight", "line"):
        return StraightTrajectory(u_ref=u_ref, duration=duration)
    if name in ("turn", "curve"):
        return TurnTrajectory(u_ref=u_ref, turn_rate=0.015, duration=duration)
    if name in ("zigzag", "zz"):
        return ZigzagTrajectory(u_ref=u_ref, duration=duration)
    raise ValueError(f"Unknown scenario: {name}")
