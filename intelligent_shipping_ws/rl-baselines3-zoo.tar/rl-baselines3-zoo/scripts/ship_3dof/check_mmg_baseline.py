from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from custom_envs.ship_3dof.config import MMGParameters
from custom_envs.ship_3dof.identification import preprocess_trial, read_trial_csv, validate_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate known MMG params against replay trials (M1 gate).")
    parser.add_argument("--mmg-params", required=True, help="Path to JSON file with MMG params.")
    parser.add_argument("--trials", nargs="+", required=True, help="CSV files with t,x,y,psi,u,v,r,delta,n columns.")
    parser.add_argument("--dt", type=float, default=0.1, help="Resampling step in seconds.")
    parser.add_argument(
        "--acceptance",
        default="scripts/ship_3dof/sim2real_acceptance.json",
        help="Acceptance config with M1 criteria.",
    )
    parser.add_argument("--output", default="artifacts/ship_3dof/mmg_baseline_report.json")
    return parser.parse_args()


def _load_mmg_params(path: str) -> MMGParameters:
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, dict) and "mmg_params" in payload and isinstance(payload["mmg_params"], dict):
        payload = payload["mmg_params"]
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid MMG payload in {path}, expected a JSON object.")
    return MMGParameters(**payload)


def main() -> None:
    args = parse_args()
    params = _load_mmg_params(args.mmg_params)
    trials = [preprocess_trial(read_trial_csv(path), dt=args.dt)[0] for path in args.trials]
    metrics = validate_parameters(trials, params=params, dt=args.dt)

    acceptance_payload = json.loads(Path(args.acceptance).read_text())
    m1_criteria = acceptance_payload["M1"]["criteria"]
    passed = (
        metrics["rmse_psi_deg"] <= float(m1_criteria["rmse_psi_deg_max"])
        and metrics["rmse_r"] <= float(m1_criteria["rmse_r_max"])
    )

    report = {
        "trial_count": len(trials),
        "metrics": metrics,
        "acceptance": {
            "m1_rmse_psi_deg_max": float(m1_criteria["rmse_psi_deg_max"]),
            "m1_rmse_r_max": float(m1_criteria["rmse_r_max"]),
            "m1_passed": passed,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
