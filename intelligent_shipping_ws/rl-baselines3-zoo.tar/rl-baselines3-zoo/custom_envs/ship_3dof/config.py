from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class MMGParameters:
    m_u: float = 35.0
    m_v: float = 45.0
    i_z: float = 18.0
    x_u: float = -6.5
    x_uu: float = -2.2
    x_n: float = 1.4
    y_v: float = -14.0
    y_r: float = 3.5
    y_vv: float = -20.0
    y_delta: float = 22.0
    n_v: float = -2.0
    n_r: float = -7.5
    n_rr: float = -5.5
    n_delta: float = 13.0


@dataclass
class ActuatorLimits:
    delta_max_deg: float = 35.0
    dot_delta_max_deg_s: float = 4.0
    n_min: float = 0.0
    n_max: float = 25.0
    dot_n_max: float = 2.0


@dataclass
class RewardWeights:
    w_progress: float = 0.85
    w_cross_track_abs: float = 1.1
    w_cross_track_sq: float = 0.35
    w_heading_abs: float = 0.6
    w_yaw_rate_sq: float = 0.08
    w_smooth_delta: float = 0.03
    w_smooth_n: float = 0.015
    w_energy: float = 0.005
    w_collision: float = 3.0
    w_out_of_channel: float = 4.0
    r_goal: float = 50.0
    r_fail: float = 100.0


@dataclass
class EnvironmentConfig:
    dt: float = 0.1
    max_steps: int = 3000
    u_min_for_beta: float = 0.25
    channel_half_width: float = 8.0
    fail_cross_track: float = 12.0
    max_abs_yaw_rate: float = 1.25
    min_forward_speed: float = -0.5
    max_forward_speed: float = 4.0
    path_length: float = 1000.0
    waypoint_step: float = 1.0
    path_curvature: float = 0.0025
    disturbance_scale: float = 0.25
    sensor_noise_std: Dict[str, float] = field(
        default_factory=lambda: {
            "e_y": 0.03,
            "e_psi": 0.01,
            "u": 0.015,
            "v": 0.015,
            "r": 0.01,
        }
    )
    actuator_lag: float = 0.2


DEFAULT_OBS_SCALE = {
    "e_y": 12.0,
    "e_psi": 3.141592653589793,
    "e_s": 250.0,
    "u": 4.0,
    "v": 2.0,
    "r": 1.5,
    "beta": 1.0,
    "delta": 0.7,
    "n": 25.0,
    "dot_delta_prev": 0.2,
    "dot_n_prev": 2.5,
    "kappa_l": 0.01,
    "d_u_hat": 0.6,
    "d_v_hat": 0.6,
}
