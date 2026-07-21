from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from custom_envs.ship_3dof.config import (
    DEFAULT_OBS_SCALE,
    ActuatorLimits,
    EnvironmentConfig,
    MMGParameters,
    RewardWeights,
)
from custom_envs.ship_3dof.dynamics import MMGDynamics, ResidualModel, ShipState
from custom_envs.ship_3dof.path import ReferencePath, wrap_to_pi


class ShipPathTracking3DOFEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        reward: dict[str, float] | None = None,
        mmg_params: dict[str, float] | None = None,
        mmg_params_path: str | None = None,
        curriculum_stage: int | None = None,
        residual_model_path: str | None = None,
        obstacles: list[tuple[float, float, float]] | None = None,
    ):
        super().__init__()
        self.cfg = EnvironmentConfig(**(config or {}))
        self.reward_cfg = RewardWeights(**(reward or {}))
        self.actuator_limits = ActuatorLimits()
        params = self._resolve_mmg_params(mmg_params=mmg_params, mmg_params_path=mmg_params_path)
        self.residual_model = ResidualModel()
        if residual_model_path is not None:
            self.residual_model.load(residual_model_path)
        self.dynamics = MMGDynamics(
            params=params,
            actuator_limits=self.actuator_limits,
            residual_model=self.residual_model,
        )
        self.curriculum_stage = curriculum_stage
        self.obstacles = obstacles or []

        self.delta_max = np.deg2rad(self.actuator_limits.delta_max_deg)
        self.dot_delta_max = np.deg2rad(self.actuator_limits.dot_delta_max_deg_s)
        self.dot_n_max = self.actuator_limits.dot_n_max
        self.dt = self.cfg.dt

        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(14,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.state = ShipState()
        self.path: ReferencePath | None = None
        self.rng = np.random.default_rng()
        self.step_count = 0
        self.episode_count = 0
        self._stage = 0
        self._last_progress = 0.0
        self._dot_delta_prev = 0.0
        self._dot_n_prev = 0.0
        self._cmd_delta = 0.0
        self._cmd_n = 0.0
        self._disturbance_body = np.zeros(2, dtype=np.float64)
        self._disturbance_hat = np.zeros(2, dtype=np.float64)

    @staticmethod
    def _resolve_mmg_params(
        mmg_params: dict[str, float] | None,
        mmg_params_path: str | None,
    ) -> MMGParameters:
        if mmg_params_path is not None:
            payload = json.loads(Path(mmg_params_path).read_text())
            if isinstance(payload, dict) and "mmg_params" in payload and isinstance(payload["mmg_params"], dict):
                payload = payload["mmg_params"]
            if not isinstance(payload, dict):
                raise ValueError(f"Invalid MMG payload in {mmg_params_path}, expected a JSON object.")
            return MMGParameters(**payload)
        return MMGParameters(**(mmg_params or {}))

    def _select_stage(self) -> int:
        if self.curriculum_stage is not None:
            return int(np.clip(self.curriculum_stage, 0, 2))
        if self.episode_count < 300:
            return 0
        if self.episode_count < 900:
            return 1
        return 2

    def _setup_episode_randomization(self) -> None:
        randomization = {0: 0.03, 1: 0.08, 2: 0.15}[self._stage]
        self.dynamics.apply_domain_randomization(self.rng, amplitude=randomization)
        disturbance_scale = self.cfg.disturbance_scale * {0: 0.0, 1: 0.55, 2: 1.0}[self._stage]
        if self._stage == 0:
            self._disturbance_body[:] = 0.0
        else:
            self._disturbance_body = self.rng.uniform(-disturbance_scale, disturbance_scale, size=2)
        estimation_noise = {0: 0.01, 1: 0.025, 2: 0.05}[self._stage]
        self._disturbance_hat = self._disturbance_body + self.rng.normal(0.0, estimation_noise, size=2)

    def _build_path(self) -> None:
        curvature_scale = {0: 0.65, 1: 0.85, 2: 1.0}[self._stage]
        curvature = self.cfg.path_curvature * curvature_scale * self.rng.uniform(0.85, 1.15)
        self.path = ReferencePath(
            length=self.cfg.path_length,
            ds=self.cfg.waypoint_step,
            curvature=curvature,
            rng=self.rng,
        )

    def _reset_state(self) -> None:
        assert self.path is not None
        # 船必须出生在路径起点附近：以路径在 s=0 处的横向位置为基准叠加噪声，
        # 否则路径横向偏移随机（±20m 量级）时回合会以 |e_y|>12 立即终止，无法学习
        y0 = float(self.path.y_at_s(0.0))
        init_heading = self.path.closest_point(0.0, y0).heading
        stage_init = {
            0: dict(y_std=0.2, psi_std_deg=1.0, u_min=0.85, u_max=1.15, v_abs=0.03, r_abs=0.02, n_min=6.0, n_max=8.0),
            1: dict(y_std=0.35, psi_std_deg=1.6, u_min=0.75, u_max=1.2, v_abs=0.04, r_abs=0.025, n_min=5.5, n_max=8.0),
            2: dict(y_std=0.5, psi_std_deg=2.0, u_min=0.6, u_max=1.2, v_abs=0.05, r_abs=0.03, n_min=5.0, n_max=8.0),
        }[self._stage]
        self.state = ShipState(
            x=0.0,
            y=float(y0 + self.rng.normal(0.0, stage_init["y_std"])),
            psi=float(init_heading + self.rng.normal(0.0, np.deg2rad(stage_init["psi_std_deg"]))),
            u=float(self.rng.uniform(stage_init["u_min"], stage_init["u_max"])),
            v=float(self.rng.uniform(-stage_init["v_abs"], stage_init["v_abs"])),
            r=float(self.rng.uniform(-stage_init["r_abs"], stage_init["r_abs"])),
            delta=0.0,
            n=float(self.rng.uniform(stage_init["n_min"], stage_init["n_max"])),
        )
        self._dot_delta_prev = 0.0
        self._dot_n_prev = 0.0
        self._cmd_delta = 0.0
        self._cmd_n = 0.0
        self.step_count = 0
        self._last_progress = 0.0

    def _tracking_error(self) -> tuple[float, float, float, float]:
        assert self.path is not None
        point = self.path.closest_point(self.state.x, self.state.y)
        e_y = self.path.signed_cross_track_error(self.state.x, self.state.y, point)
        e_psi = wrap_to_pi(self.state.psi - point.heading)
        e_s = self.path.final_s - point.s
        return e_y, e_psi, e_s, point.curvature

    def _sensor_noise(self, key: str) -> float:
        if self._stage < 2:
            return 0.0
        return float(self.rng.normal(0.0, self.cfg.sensor_noise_std[key]))

    def _observation(self) -> np.ndarray:
        e_y, e_psi, e_s, kappa_l = self._tracking_error()
        beta = float(np.arctan2(self.state.v, max(self.state.u, self.cfg.u_min_for_beta)))
        obs_raw = np.array(
            [
                e_y + self._sensor_noise("e_y"),
                e_psi + self._sensor_noise("e_psi"),
                e_s,
                self.state.u + self._sensor_noise("u"),
                self.state.v + self._sensor_noise("v"),
                self.state.r + self._sensor_noise("r"),
                beta,
                self.state.delta,
                self.state.n,
                self._dot_delta_prev,
                self._dot_n_prev,
                kappa_l,
                self._disturbance_hat[0],
                self._disturbance_hat[1],
            ],
            dtype=np.float64,
        )
        scale = np.array(list(DEFAULT_OBS_SCALE.values()), dtype=np.float64)
        return np.clip(obs_raw / scale, -5.0, 5.0).astype(np.float32)

    def _collision_penalty(self) -> tuple[float, bool]:
        for ox, oy, radius in self.obstacles:
            if (self.state.x - ox) ** 2 + (self.state.y - oy) ** 2 < radius**2:
                return self.reward_cfg.w_collision, True
        return 0.0, False

    def _reward(self, dot_delta_cmd: float, dot_n_cmd: float) -> tuple[float, dict[str, float], bool, bool]:
        assert self.path is not None
        point = self.path.closest_point(self.state.x, self.state.y)
        delta_s = max(0.0, point.s - self._last_progress)
        self._last_progress = point.s
        e_y = self.path.signed_cross_track_error(self.state.x, self.state.y, point)
        e_psi = wrap_to_pi(self.state.psi - point.heading)

        r_progress = self.reward_cfg.w_progress * np.clip(delta_s, 0.0, 1.25)
        r_track = self.reward_cfg.w_cross_track_abs * abs(e_y) + self.reward_cfg.w_cross_track_sq * e_y**2
        r_heading = self.reward_cfg.w_heading_abs * abs(e_psi) + self.reward_cfg.w_yaw_rate_sq * self.state.r**2
        r_smooth = self.reward_cfg.w_smooth_delta * dot_delta_cmd**2 + self.reward_cfg.w_smooth_n * dot_n_cmd**2
        r_energy = self.reward_cfg.w_energy * self.state.n**2
        collision_penalty, collision = self._collision_penalty()
        out_of_channel = abs(e_y) > self.cfg.channel_half_width
        r_safety = collision_penalty + (self.reward_cfg.w_out_of_channel if out_of_channel else 0.0)
        reward = float(r_progress - r_track - r_heading - r_smooth - r_energy - r_safety)
        terms = {
            "r_progress": float(r_progress),
            "r_track": float(r_track),
            "r_heading": float(r_heading),
            "r_smooth": float(r_smooth),
            "r_energy": float(r_energy),
            "r_safety": float(r_safety),
            "delta_s": float(delta_s),
        }
        return reward, terms, collision, out_of_channel

    def _termination(self, collision: bool, out_of_channel: bool) -> tuple[bool, bool, bool]:
        assert self.path is not None
        point = self.path.closest_point(self.state.x, self.state.y)
        goal = point.s >= self.path.final_s - 1.0
        fail = (
            collision
            or abs(self.path.signed_cross_track_error(self.state.x, self.state.y, point)) > self.cfg.fail_cross_track
            or abs(self.state.r) > self.cfg.max_abs_yaw_rate
            or self.state.u < self.cfg.min_forward_speed
            or self.state.u > self.cfg.max_forward_speed
            or out_of_channel
        )
        terminated = goal or fail
        truncated = self.step_count >= self.cfg.max_steps
        return terminated, truncated, goal

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.episode_count += 1
        self._stage = self._select_stage()
        self._setup_episode_randomization()
        self._build_path()
        self._reset_state()
        observation = self._observation()
        info = {
            "curriculum_stage": self._stage,
            "mmg_params": asdict(self.dynamics.params),
        }
        return observation, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, -1.0, 1.0)
        dot_delta_cmd = float(action[0] * self.dot_delta_max)
        dot_n_cmd = float(action[1] * self.dot_n_max)

        alpha = self.dt / (self.cfg.actuator_lag + self.dt)
        if self._stage >= 2:
            self._cmd_delta = (1.0 - alpha) * self._cmd_delta + alpha * dot_delta_cmd
            self._cmd_n = (1.0 - alpha) * self._cmd_n + alpha * dot_n_cmd
        else:
            self._cmd_delta = dot_delta_cmd
            self._cmd_n = dot_n_cmd

        disturbance = self._disturbance_body.copy()
        if self._stage >= 2:
            drift = self.rng.normal(0.0, 0.01, size=2)
            self._disturbance_body = np.clip(self._disturbance_body + drift, -0.6, 0.6)
            disturbance = self._disturbance_body.copy()

        self.state = self.dynamics.step(
            state=self.state,
            dot_delta_cmd=float(self._cmd_delta),
            dot_n_cmd=float(self._cmd_n),
            dt=self.dt,
            disturbance_body=(float(disturbance[0]), float(disturbance[1])),
        )
        self.step_count += 1
        self._dot_delta_prev = float(self._cmd_delta)
        self._dot_n_prev = float(self._cmd_n)

        reward, reward_terms, collision, out_of_channel = self._reward(self._cmd_delta, self._cmd_n)
        terminated, truncated, goal = self._termination(collision, out_of_channel)
        if goal:
            reward += self.reward_cfg.r_goal
        if terminated and not goal:
            reward -= self.reward_cfg.r_fail

        observation = self._observation()
        info = {
            "reward_terms": reward_terms,
            "curriculum_stage": self._stage,
            "collision": collision,
            "out_of_channel": out_of_channel,
            "success": goal,
        }
        return observation, float(reward), terminated, truncated, info
