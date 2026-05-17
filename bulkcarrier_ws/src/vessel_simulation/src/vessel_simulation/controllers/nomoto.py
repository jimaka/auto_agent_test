"""Nomoto-style fallback controller for SIL baseline."""

from __future__ import annotations

from typing import List

import numpy as np

from vessel_simulation.metrics import cross_track_error, heading_error, speed_error
from vessel_simulation.mmg3dof import ControlInput, VesselState


class NomotoController:
    def __init__(self, K: float = 0.8, T: float = 20.0, k_rpm: float = 2.0) -> None:
        self.K = K
        self.T = T
        self.k_rpm = k_rpm
        self.delta_max = np.deg2rad(35.0)
        self.n_bounds = (0.0, 120.0)

    def compute(self, state: VesselState, u_prev: ControlInput, x_refs: List[VesselState]) -> ControlInput:
        ref = x_refs[0] if x_refs else state
        e_psi = heading_error(state, ref)
        e_cross = cross_track_error(state, ref)
        e_u = speed_error(state, ref)

        delta = -self.K * e_psi - 0.2 * e_cross
        delta = float(np.clip(delta, -self.delta_max, self.delta_max))

        rpm = u_prev.rpm + self.k_rpm * e_u
        rpm = float(np.clip(rpm, self.n_bounds[0], self.n_bounds[1]))
        return ControlInput(delta_rad=delta, rpm=rpm)
