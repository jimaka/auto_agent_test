from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SafetyLimits:
    delta_max: float
    dot_delta_max: float
    n_min: float
    n_max: float
    dot_n_max: float
    max_abs_r: float
    max_abs_e_y: float


class SafetyFilter:
    def __init__(self, limits: SafetyLimits, dt: float):
        self.limits = limits
        self.dt = dt

    def filter_action(
        self,
        action: np.ndarray,
        current_delta: float,
        current_n: float,
        estimated_r: float,
        estimated_e_y: float,
    ) -> tuple[np.ndarray, bool]:
        safe_action = np.clip(action, -1.0, 1.0).astype(np.float64)
        dot_delta_cmd = safe_action[0] * self.limits.dot_delta_max
        dot_n_cmd = safe_action[1] * self.limits.dot_n_max
        proposed_delta = np.clip(current_delta + dot_delta_cmd * self.dt, -self.limits.delta_max, self.limits.delta_max)
        proposed_n = np.clip(current_n + dot_n_cmd * self.dt, self.limits.n_min, self.limits.n_max)
        violated = abs(estimated_r) > self.limits.max_abs_r or abs(estimated_e_y) > self.limits.max_abs_e_y
        if violated:
            return np.array([0.0, 0.0], dtype=np.float64), True
        corrected = np.array(
            [
                (proposed_delta - current_delta) / max(self.limits.dot_delta_max * self.dt, 1e-6),
                (proposed_n - current_n) / max(self.limits.dot_n_max * self.dt, 1e-6),
            ],
            dtype=np.float64,
        )
        return np.clip(corrected, -1.0, 1.0), False


class FallbackPIDController:
    def __init__(
        self,
        kp_ey: float = 0.25,
        kp_epsi: float = 0.9,
        kd_r: float = 0.4,
        target_n: float = 8.0,
    ):
        self.kp_ey = kp_ey
        self.kp_epsi = kp_epsi
        self.kd_r = kd_r
        self.target_n = target_n

    def action(
        self,
        e_y: float,
        e_psi: float,
        r: float,
        current_n: float,
        dot_delta_max: float,
        dot_n_max: float,
    ) -> np.ndarray:
        dot_delta = -self.kp_ey * e_y - self.kp_epsi * e_psi - self.kd_r * r
        dot_delta = np.clip(dot_delta, -dot_delta_max, dot_delta_max)
        dot_n = np.clip(self.target_n - current_n, -dot_n_max, dot_n_max)
        return np.array([dot_delta / max(dot_delta_max, 1e-6), dot_n / max(dot_n_max, 1e-6)], dtype=np.float64)


class ShadowModeSupervisor:
    def __init__(self):
        self.shadow_decisions = 0
        self.takeover_recommendations = 0

    def observe(self, filtered_override: bool) -> None:
        self.shadow_decisions += 1
        if filtered_override:
            self.takeover_recommendations += 1

    def recommendation_rate(self) -> float:
        if self.shadow_decisions == 0:
            return 0.0
        return self.takeover_recommendations / self.shadow_decisions
