from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from custom_envs.ship_3dof.config import MMGParameters
from custom_envs.ship_3dof.dynamics import MMGDynamics, ShipState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic MMG trial CSV files for pipeline bootstrap.")
    parser.add_argument("--mmg-params", required=True, help="Path to JSON with MMG parameters.")
    parser.add_argument("--output-dir", default="artifacts/ship_3dof/trials")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--duration", type=float, default=180.0)
    return parser.parse_args()


def _load_mmg_params(path: str) -> MMGParameters:
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, dict) and "mmg_params" in payload and isinstance(payload["mmg_params"], dict):
        payload = payload["mmg_params"]
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid MMG payload in {path}, expected a JSON object.")
    return MMGParameters(**payload)


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["t", "x", "y", "psi", "u", "v", "r", "delta", "n"])
        writer.writeheader()
        writer.writerows(rows)


def _simulate_trial(
    dynamics: MMGDynamics,
    dt: float,
    duration: float,
    command_fn,
) -> list[dict[str, float]]:
    steps = int(duration / dt)
    state = ShipState(u=1.0, n=8.0)
    rows: list[dict[str, float]] = []
    for i in range(steps + 1):
        t = i * dt
        rows.append(
            {
                "t": float(t),
                "x": float(state.x),
                "y": float(state.y),
                "psi": float(state.psi),
                "u": float(state.u),
                "v": float(state.v),
                "r": float(state.r),
                "delta": float(state.delta),
                "n": float(state.n),
            }
        )
        dot_delta, dot_n = command_fn(state, t, dt)
        state = dynamics.step(state, dot_delta_cmd=dot_delta, dot_n_cmd=dot_n, dt=dt, disturbance_body=(0.0, 0.0))
    return rows


def _make_turn_command(target_delta_deg: float, target_n: float):
    target_delta = np.deg2rad(target_delta_deg)

    def command(state: ShipState, _: float, dt: float) -> tuple[float, float]:
        dot_delta = (target_delta - state.delta) / max(dt, 1e-6)
        dot_n = (target_n - state.n) / max(dt, 1e-6)
        return float(dot_delta), float(dot_n)

    return command


def _make_zigzag_command(target_delta_deg: float, psi_switch_deg: float, target_n: float):
    target_delta = np.deg2rad(target_delta_deg)
    psi_switch = np.deg2rad(psi_switch_deg)
    sign = 1.0

    def command(state: ShipState, _: float, dt: float) -> tuple[float, float]:
        nonlocal sign
        if sign > 0 and state.psi >= psi_switch:
            sign = -1.0
        elif sign < 0 and state.psi <= -psi_switch:
            sign = 1.0
        dot_delta = (sign * target_delta - state.delta) / max(dt, 1e-6)
        dot_n = (target_n - state.n) / max(dt, 1e-6)
        return float(dot_delta), float(dot_n)

    return command


def main() -> None:
    args = parse_args()
    params = _load_mmg_params(args.mmg_params)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def make_dynamics() -> MMGDynamics:
        return MMGDynamics(params=MMGParameters(**vars(params)))

    scenarios = {
        "turn_port.csv": _make_turn_command(target_delta_deg=10.0, target_n=8.0),
        "turn_starboard.csv": _make_turn_command(target_delta_deg=-10.0, target_n=8.0),
        "zigzag_10_10.csv": _make_zigzag_command(target_delta_deg=10.0, psi_switch_deg=10.0, target_n=8.0),
    }
    for filename, command_fn in scenarios.items():
        rows = _simulate_trial(make_dynamics(), dt=args.dt, duration=args.duration, command_fn=command_fn)
        _write_csv(output_dir / filename, rows)

    print(f"Generated {len(scenarios)} synthetic trials at {output_dir}")


if __name__ == "__main__":
    main()
