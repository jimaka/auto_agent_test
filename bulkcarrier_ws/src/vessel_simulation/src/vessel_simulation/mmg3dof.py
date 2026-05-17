"""3-DOF MMG-style maneuvering model for bulk carrier SIL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import yaml


@dataclass
class VesselState:
    x: float
    y: float
    psi: float
    u: float
    v: float
    r: float

    def as_vector(self) -> np.ndarray:
        return np.array([self.x, self.y, self.psi, self.u, self.v, self.r], dtype=np.float64)

    @classmethod
    def from_vector(cls, v: np.ndarray) -> "VesselState":
        return cls(float(v[0]), float(v[1]), float(v[2]), float(v[3]), float(v[4]), float(v[5]))


@dataclass
class ControlInput:
    delta_rad: float
    rpm: float


def wrap_pi(a: float) -> float:
    while a > np.pi:
        a -= 2.0 * np.pi
    while a < -np.pi:
        a += 2.0 * np.pi
    return a


class MMG3DOF:
    """
    Simplified 3-DOF model (surge/sway/yaw) with rudder and thrust.
    Velocities in body frame; position in local NED-like frame.
    """

    def __init__(self, params: Dict[str, Any], dt: float = 0.05) -> None:
        self.dt = dt
        ship = params.get("ship", {})
        hydro = params.get("hydro", {})
        act = params.get("actuator", {})

        self.Lpp = float(ship.get("Lpp", 65.0))
        self.m = float(ship.get("m", 4.2e6))
        self.Iz = float(ship.get("Iz", 3.5e9))

        self.Xu = float(hydro.get("Xu", -12000.0))
        self.Yv = float(hydro.get("Yv", -80000.0))
        self.Nr = float(hydro.get("Nr", -8.0e6))
        self.Yr = float(hydro.get("Yr", -2.0e6))
        self.Nv = float(hydro.get("Nv", -2.0e6))
        self.Ndelta = float(hydro.get("Ndelta", -4.0e6))
        self.Xthrust = float(hydro.get("Xthrust", 8.0))
        self.Yvv = float(hydro.get("Yvv", -2000.0))
        self.Nrr = float(hydro.get("Nrr", -1.0e8))

        self.delta_max = np.deg2rad(float(act.get("delta_max_deg", 35.0)))
        self.n_max = float(act.get("n_max", 120.0))

    @classmethod
    def from_yaml(cls, path: str) -> "MMG3DOF":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        dt = float(raw.get("mmg", {}).get("dt", 0.05))
        return cls(raw, dt=dt)

    def _dynamics(self, state: VesselState, ctrl: ControlInput) -> np.ndarray:
        u, v, r = state.u, state.v, state.r
        delta = float(np.clip(ctrl.delta_rad, -self.delta_max, self.delta_max))
        thrust = self.Xthrust * ctrl.rpm

        u_dot = (thrust + self.Xu * u) / self.m
        v_dot = (self.Yv * v + self.Yr * r + self.Yvv * abs(v) * v) / self.m
        r_dot = (self.Nr * r + self.Nv * v + self.Ndelta * delta + self.Nrr * abs(r) * r) / self.Iz

        x_dot = u * np.cos(state.psi) - v * np.sin(state.psi)
        y_dot = u * np.sin(state.psi) + v * np.cos(state.psi)
        psi_dot = r
        return np.array([x_dot, y_dot, psi_dot, u_dot, v_dot, r_dot], dtype=np.float64)

    def step(self, state: VesselState, ctrl: ControlInput) -> VesselState:
        return self._rk4(state, ctrl)

    def _rk4(self, state: VesselState, ctrl: ControlInput) -> VesselState:
        x0 = state.as_vector()

        def f(xv: np.ndarray) -> np.ndarray:
            return self._dynamics(VesselState.from_vector(xv), ctrl)

        k1 = f(x0)
        k2 = f(x0 + 0.5 * self.dt * k1)
        k3 = f(x0 + 0.5 * self.dt * k2)
        k4 = f(x0 + self.dt * k3)
        xn = x0 + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        xn[2] = wrap_pi(xn[2])
        return VesselState.from_vector(xn)
