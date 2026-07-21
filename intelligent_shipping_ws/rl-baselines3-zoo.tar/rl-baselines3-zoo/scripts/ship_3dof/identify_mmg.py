from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from custom_envs.ship_3dof.config import MMGParameters
from custom_envs.ship_3dof.identification import (
    IdentificationResult,
    fit_mmg_least_squares,
    fit_nomoto,
    preprocess_trial,
    read_trial_csv,
    refine_mmg_multiple_shooting,
    save_identification_result,
    validate_parameters,
)


def _split_trials(paths: list[str], train_ratio: float) -> tuple[list[str], list[str]]:
    shuffled = sorted(paths)
    split = max(1, int(len(shuffled) * train_ratio))
    if split >= len(shuffled):
        split = len(shuffled) - 1
    return shuffled[:split], shuffled[split:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Identify MMG 3-DOF parameters from sea-trial CSV files.")
    parser.add_argument("--trials", nargs="+", required=True, help="CSV files with t,x,y,psi,u,v,r,delta,n columns.")
    parser.add_argument("--dt", type=float, default=0.1, help="Resampling step in seconds.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Fraction of files used for fitting.")
    parser.add_argument(
        "--masses",
        nargs=3,
        type=float,
        default=(35.0, 45.0, 18.0),
        metavar=("M_U", "M_V", "I_Z"),
        help="Fixed masses used in linearized MMG regression.",
    )
    parser.add_argument("--output", type=str, default="artifacts/ship_3dof/identification_result.json")
    parser.add_argument("--params-yaml", type=str, default="artifacts/ship_3dof/mmg_params.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.trials) < 2:
        raise ValueError("Need at least two trial files to keep a holdout split.")

    train_paths, val_paths = _split_trials(args.trials, args.train_ratio)
    train_processed = []
    for path in train_paths:
        trial = read_trial_csv(path)
        train_processed.append(preprocess_trial(trial, dt=args.dt))
    val_trials = [preprocess_trial(read_trial_csv(path), dt=args.dt)[0] for path in val_paths]

    nomoto_k, nomoto_t = fit_nomoto(train_processed)
    params = fit_mmg_least_squares(train_processed, masses=tuple(args.masses))
    train_trials = [trial for trial, _ in train_processed]
    params = refine_mmg_multiple_shooting(train_trials, params, dt=args.dt)
    validation = validate_parameters(val_trials, params, dt=args.dt)

    result = IdentificationResult(
        nomoto_k=nomoto_k,
        nomoto_t=nomoto_t,
        mmg_params=params,
        validation=validation,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_identification_result(output_path, result)

    params_path = Path(args.params_yaml)
    params_path.parent.mkdir(parents=True, exist_ok=True)
    params_path.write_text(json.dumps({"mmg_params": vars(params), "nomoto": {"k": nomoto_k, "t": nomoto_t}}, indent=2))

    print("Identification finished")
    print(f"Nomoto K={nomoto_k:.4f}, T={nomoto_t:.4f}")
    print(f"Validation: {validation}")
    if validation["rmse_psi_deg"] > 5.0:
        print("Warning: rmse_psi_deg exceeds suggested 3-5 deg target.")
    if validation["rmse_r"] > 0.1:
        print("Warning: rmse_r exceeds suggested 10% target for normalized yaw-rate range.")


if __name__ == "__main__":
    main()
