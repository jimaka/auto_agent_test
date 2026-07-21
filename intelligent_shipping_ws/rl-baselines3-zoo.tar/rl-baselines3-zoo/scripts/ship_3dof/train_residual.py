from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from custom_envs.ship_3dof.config import MMGParameters
from custom_envs.ship_3dof.dynamics import MMGDynamics, ResidualModel, ShipState
from custom_envs.ship_3dof.identification import preprocess_trial, read_trial_csv


class ResidualMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 3),
            nn.Tanh(),
        )
        self.scale = nn.Parameter(torch.tensor([0.08, 0.08, 0.05], dtype=torch.float32), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) * self.scale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train residual model eps_hat for MMG dynamics.")
    parser.add_argument("--trials", nargs="+", required=True)
    parser.add_argument("--mmg-params", type=str, required=True, help="JSON file containing mmg_params dict.")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="artifacts/ship_3dof/residual_model.npz")
    parser.add_argument("--metrics-output", type=str, default="artifacts/ship_3dof/residual_metrics.json")
    return parser.parse_args()


def build_dataset(trials: list[str], params: MMGParameters, dt: float) -> tuple[np.ndarray, np.ndarray]:
    dynamics = MMGDynamics(params=params)
    features = []
    targets = []
    for path in trials:
        trial, _ = preprocess_trial(read_trial_csv(path), dt=dt)
        for i in range(len(trial.t) - 1):
            state = ShipState(
                x=float(trial.x[i]),
                y=float(trial.y[i]),
                psi=float(trial.psi[i]),
                u=float(trial.u[i]),
                v=float(trial.v[i]),
                r=float(trial.r[i]),
                delta=float(trial.delta[i]),
                n=float(trial.n[i]),
            )
            dot_delta = (trial.delta[i + 1] - trial.delta[i]) / dt
            dot_n = (trial.n[i + 1] - trial.n[i]) / dt
            pred = ShipState(**vars(state))
            pred = dynamics.step(pred, dot_delta_cmd=dot_delta, dot_n_cmd=dot_n, dt=dt, disturbance_body=(0.0, 0.0))
            eps = np.array(
                [
                    (trial.u[i + 1] - pred.u) / dt,
                    (trial.v[i + 1] - pred.v) / dt,
                    (trial.r[i + 1] - pred.r) / dt,
                ],
                dtype=np.float64,
            )
            features.append([state.u, state.v, state.r, state.delta, state.n])
            targets.append(eps)
    return np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def save_residual_model(model: ResidualMLP, path: str) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    residual = ResidualModel(hidden_size=64)
    residual.w1 = model.net[0].weight.detach().cpu().numpy().T
    residual.b1 = model.net[0].bias.detach().cpu().numpy()
    residual.w2 = model.net[2].weight.detach().cpu().numpy().T
    residual.b2 = model.net[2].bias.detach().cpu().numpy()
    residual.w3 = model.net[4].weight.detach().cpu().numpy().T
    residual.b3 = model.net[4].bias.detach().cpu().numpy()
    residual.scale = model.scale.detach().cpu().numpy()
    residual.save(path_obj)


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.mmg_params).read_text())
    params = MMGParameters(**payload["mmg_params"])
    x, y = build_dataset(args.trials, params=params, dt=args.dt)
    if len(x) < 16:
        raise ValueError("Residual training needs at least 16 samples.")

    if not 0.0 < args.validation_split < 0.5:
        raise ValueError("--validation-split must be in (0, 0.5).")
    rng = np.random.default_rng(args.seed)
    indices = np.arange(len(x))
    rng.shuffle(indices)
    val_size = max(1, int(len(indices) * args.validation_split))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    if len(train_idx) == 0:
        raise ValueError("Validation split too large, train set is empty.")

    x_train = torch.from_numpy(x[train_idx])
    y_train = torch.from_numpy(y[train_idx])
    x_val = torch.from_numpy(x[val_idx])
    y_val = torch.from_numpy(y[val_idx])

    ds = TensorDataset(x_train, y_train)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    model = ResidualMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    baseline_val_mse = float(torch.mean(y_val**2).item())

    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(args.epochs):
        losses = []
        for xb, yb in loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        with torch.no_grad():
            val_pred = model(x_val)
            val_loss = float(criterion(val_pred, y_val).item())
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"epoch={epoch + 1:03d} train_loss={np.mean(losses):.6f} val_loss={val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    save_residual_model(model, args.output)
    validation_loss_reduction = max(0.0, (baseline_val_mse - best_val_loss) / max(baseline_val_mse, 1e-8))
    metrics = {
        "samples_total": int(len(x)),
        "samples_train": int(len(train_idx)),
        "samples_validation": int(len(val_idx)),
        "baseline_val_mse": baseline_val_mse,
        "best_val_mse": best_val_loss,
        "validation_loss_reduction": validation_loss_reduction,
    }
    metrics_path = Path(args.metrics_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Saved residual model: {args.output}")
    print(f"Saved residual metrics: {args.metrics_output}")


if __name__ == "__main__":
    main()
