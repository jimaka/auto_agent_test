from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def wrap_to_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass
class PathPoint:
    x: float
    y: float
    heading: float
    curvature: float
    s: float


class ReferencePath:
    def __init__(self, length: float, ds: float, curvature: float, rng: np.random.Generator):
        self.length = length
        self.ds = ds
        self.curvature = curvature
        self.rng = rng
        self._s = np.arange(0.0, length + ds, ds, dtype=np.float64)
        self._x, self._y = self._build_points(self._s)
        self._heading = self._compute_heading(self._x, self._y)
        self._kappa = self._compute_curvature(self._x, self._y)
        self._last_closest = 0

    def _build_points(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        phase_1 = self.rng.uniform(0.0, 2.0 * np.pi)
        phase_2 = self.rng.uniform(0.0, 2.0 * np.pi)
        y = 12.0 * self.curvature * self.length * np.sin(2.0 * np.pi * s / self.length + phase_1)
        y += 6.0 * self.curvature * self.length * np.sin(4.0 * np.pi * s / self.length + phase_2)
        return s.copy(), y

    @staticmethod
    def _compute_heading(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        dx = np.gradient(x)
        dy = np.gradient(y)
        return np.arctan2(dy, dx)

    @staticmethod
    def _compute_curvature(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        denom = np.maximum((dx**2 + dy**2) ** 1.5, 1e-6)
        return (dx * ddy - dy * ddx) / denom

    @property
    def final_s(self) -> float:
        return float(self._s[-1])

    def y_at_s(self, s: float) -> float:
        """路径在弧长 s 处的横向位置（线性插值）。"""
        return float(np.interp(s, self._s, self._y))

    def closest_point(self, x: float, y: float, look_ahead: int = 250) -> PathPoint:
        start = max(0, self._last_closest - 10)
        stop = min(len(self._s), self._last_closest + look_ahead)
        sx = self._x[start:stop]
        sy = self._y[start:stop]
        dist_sq = (sx - x) ** 2 + (sy - y) ** 2
        local_idx = int(np.argmin(dist_sq))
        idx = start + local_idx
        self._last_closest = idx
        return PathPoint(
            x=float(self._x[idx]),
            y=float(self._y[idx]),
            heading=float(self._heading[idx]),
            curvature=float(self._kappa[idx]),
            s=float(self._s[idx]),
        )

    def signed_cross_track_error(self, x: float, y: float, point: PathPoint) -> float:
        tangent = np.array([np.cos(point.heading), np.sin(point.heading)], dtype=np.float64)
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
        delta = np.array([x - point.x, y - point.y], dtype=np.float64)
        return float(np.dot(delta, normal))
