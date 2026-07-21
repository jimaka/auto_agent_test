from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from custom_envs.ship_3dof.config import ActuatorLimits, MMGParameters


@dataclass
class ShipState:
    x: float = 0.0
    y: float = 0.0
    psi: float = 0.0
    u: float = 0.0
    v: float = 0.0
    r: float = 0.0
    delta: float = 0.0
    n: float = 0.0


class ResidualModel:
    def __init__(self, hidden_size: int = 64):
        self.hidden_size = hidden_size
        self.w1 = np.zeros((5, hidden_size), dtype=np.float64)
        self.b1 = np.zeros(hidden_size, dtype=np.float64)
        self.w2 = np.zeros((hidden_size, hidden_size), dtype=np.float64)
        self.b2 = np.zeros(hidden_size, dtype=np.float64)
        self.w3 = np.zeros((hidden_size, 3), dtype=np.float64)
        self.b3 = np.zeros(3, dtype=np.float64)
        self.scale = np.array([0.08, 0.08, 0.05], dtype=np.float64)

    def load(self, path: str | Path) -> None:
        payload = np.load(path)
        self.w1 = payload["w1"]
        self.b1 = payload["b1"]
        self.w2 = payload["w2"]
        self.b2 = payload["b2"]
        self.w3 = payload["w3"]
        self.b3 = payload["b3"]
        if "scale" in payload:
            self.scale = payload["scale"]

    def save(self, path: str | Path) -> None:
        np.savez(
            path,
            w1=self.w1,
            b1=self.b1,
            w2=self.w2,
            b2=self.b2,
            w3=self.w3,
            b3=self.b3,
            scale=self.scale,
        )

    def predict(self, state: ShipState) -> np.ndarray:
        x = np.array([state.u, state.v, state.r, state.delta, state.n], dtype=np.float64)
        h1 = np.tanh(x @ self.w1 + self.b1)
        h2 = np.tanh(h1 @ self.w2 + self.b2)
        y = h2 @ self.w3 + self.b3
        return self.scale * np.tanh(y)


class MMGDynamics:
    def __init__(
        self,
        params: MMGParameters | None = None,
        actuator_limits: ActuatorLimits | None = None,
        residual_model: ResidualModel | None = None,
    ):
        self.params = params or MMGParameters()
        self.base_params = MMGParameters(**asdict(self.params))
        self.actuator_limits = actuator_limits or ActuatorLimits()
        self.residual_model = residual_model or ResidualModel()
        self.delta_max = np.deg2rad(self.actuator_limits.delta_max_deg)
        self.dot_delta_max = np.deg2rad(self.actuator_limits.dot_delta_max_deg_s)

    def set_params(self, params: MMGParameters, update_base: bool = False) -> None:
        self.params = params
        if update_base:
            self.base_params = MMGParameters(**asdict(params))

    def set_base_params(self, params: MMGParameters) -> None:
        self.base_params = MMGParameters(**asdict(params))
        self.params = MMGParameters(**asdict(params))

    def apply_domain_randomization(self, rng: np.random.Generator, amplitude: float = 0.2) -> None:
        randomized = {}
        for key, value in asdict(self.base_params).items():
            low = 1.0 - amplitude
            high = 1.0 + amplitude
            randomized[key] = float(value * rng.uniform(low, high))
        self.params = MMGParameters(**randomized)

    def _forces(self, state: ShipState) -> tuple[float, float, float]:
        p = self.params
        x_force = p.x_u * state.u + p.x_uu * abs(state.u) * state.u + p.x_n * state.n * abs(state.n)
        y_force = p.y_v * state.v + p.y_r * state.r + p.y_vv * abs(state.v) * state.v + p.y_delta * state.u**2 * state.delta
        n_moment = p.n_v * state.v + p.n_r * state.r + p.n_rr * abs(state.r) * state.r + p.n_delta * state.u**2 * state.delta
        return x_force, y_force, n_moment

    def step(
        self,
        state: ShipState,
        dot_delta_cmd: float,
        dot_n_cmd: float,
        dt: float,
        disturbance_body: tuple[float, float],
    ) -> ShipState:
        dot_delta = float(np.clip(dot_delta_cmd, -self.dot_delta_max, self.dot_delta_max))
        dot_n = float(np.clip(dot_n_cmd, -self.actuator_limits.dot_n_max, self.actuator_limits.dot_n_max))
        state.delta = float(np.clip(state.delta + dot_delta * dt, -self.delta_max, self.delta_max))
        state.n = float(np.clip(state.n + dot_n * dt, self.actuator_limits.n_min, self.actuator_limits.n_max))

        x_force, y_force, n_moment = self._forces(state)
        residual = self.residual_model.predict(state)

        u_dot = x_force / self.params.m_u + state.r * state.v + residual[0]
        v_dot = y_force / self.params.m_v - state.r * state.u + residual[1]
        r_dot = n_moment / self.params.i_z + residual[2]

        state.u += u_dot * dt
        state.v += v_dot * dt
        state.r += r_dot * dt
        state.psi += state.r * dt

        d_u, d_v = disturbance_body
        c_psi = np.cos(state.psi)
        s_psi = np.sin(state.psi)
        x_dot = c_psi * (state.u + d_u) - s_psi * (state.v + d_v)
        y_dot = s_psi * (state.u + d_u) + c_psi * (state.v + d_v)
        state.x += x_dot * dt
        state.y += y_dot * dt
        return state
