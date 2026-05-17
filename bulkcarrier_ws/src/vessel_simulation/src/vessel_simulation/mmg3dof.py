"""3-DOF MMG vessel model (surge, sway, yaw). Scaffold only."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class VesselState:
    x: float
    y: float
    psi: float
    u: float
    v: float
    r: float

    def as_vector(self) -> np.ndarray:
        return np.array([self.x, self.y, self.psi, self.u, self.v, self.r])


@dataclass
class ControlInput:
    delta_rad: float
    rpm: float


class MMG3DOF:
    """Placeholder integrator; replace with ship-specific MMG parameters."""

    def __init__(self, dt: float = 0.05) -> None:
        self.dt = dt

    def step(self, state: VesselState, u: ControlInput) -> VesselState:
        # TODO: implement MMG dynamics for 65 m bulk carrier
        x = state.as_vector()
        x_next = x.copy()
        x_next[0] += state.u * np.cos(state.psi) * self.dt
        x_next[1] += state.u * np.sin(state.psi) * self.dt
        x_next[2] += state.r * self.dt
        return VesselState(*x_next.tolist())
