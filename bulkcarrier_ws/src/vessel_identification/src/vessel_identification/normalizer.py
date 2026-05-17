"""Standardize states and inputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Normalizer:
    mu: np.ndarray
    sigma: np.ndarray
    eps: float = 1e-6

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mu) / (self.sigma + self.eps)

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return x * (self.sigma + self.eps) + self.mu

    @classmethod
    def fit(cls, data: np.ndarray, eps: float = 1e-6) -> "Normalizer":
        mu = np.mean(data, axis=0)
        sigma = np.std(data, axis=0)
        sigma = np.where(sigma < eps, 1.0, sigma)
        return cls(mu=mu, sigma=sigma, eps=eps)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(
                {"mu": self.mu.tolist(), "sigma": self.sigma.tolist()},
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: Path, eps: float = 1e-6) -> "Normalizer":
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls(mu=np.asarray(raw["mu"], dtype=np.float64),
                   sigma=np.asarray(raw["sigma"], dtype=np.float64),
                   eps=eps)
