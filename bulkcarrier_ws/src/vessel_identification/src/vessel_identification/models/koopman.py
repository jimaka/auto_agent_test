"""Deep Koopman: NN encoder + global linear (A, B)."""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "elu":
        return nn.ELU()
    raise ValueError(f"Unknown activation: {name}")


class MlpEncoder(nn.Module):
    def __init__(self, nx: int, nz: int, hidden: List[int], activation: str = "relu") -> None:
        super().__init__()
        layers: List[nn.Module] = []
        dim_in = nx
        act = _activation(activation)
        for h in hidden:
            layers.extend([nn.Linear(dim_in, h), act])
            dim_in = h
        layers.append(nn.Linear(dim_in, nz))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepKoopman(nn.Module):
    """
    z_k = phi(x_k)
    z_{k+1} = A z_k + B u_k
    x_k ~= C z_k  (linear decoder for reconstruction / Cx export)
    """

    def __init__(
        self,
        nx: int = 6,
        nu: int = 2,
        nz: int = 64,
        hidden: List[int] | None = None,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        hidden = hidden or [128, 128]
        self.nx = nx
        self.nu = nu
        self.nz = nz
        self.encoder = MlpEncoder(nx, nz, hidden, activation)
        self.decoder = nn.Linear(nz, nx, bias=True)
        self.A = nn.Parameter(torch.eye(nz) * 0.9 + torch.randn(nz, nz) * 0.01)
        self.B = nn.Parameter(torch.randn(nz, nu) * 0.01)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., nx] -> z: [..., nz]"""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def step(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """z: [B, nz], u: [B, nu] -> z_next: [B, nz]"""
        return z @ self.A.T + u @ self.B.T

    def rollout(self, z0: torch.Tensor, u_seq: torch.Tensor) -> torch.Tensor:
        """
        z0: [B, nz]
        u_seq: [B, N, nu]
        returns z_pred: [B, N+1, nz] including z0
        """
        b, n, _ = u_seq.shape
        zs = [z0]
        z = z0
        for i in range(n):
            z = self.step(z, u_seq[:, i, :])
            zs.append(z)
        return torch.stack(zs, dim=1)

    def multi_step_targets(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Encode each x_t in window -> [B, N+1, nz]."""
        return self.encode(x_seq)
