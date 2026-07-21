from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full Ship3DOF pipeline with known MMG parameters.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--mmg-params", required=True, help="Path to known MMG parameter JSON.")
    parser.add_argument("--trials", nargs="+", default=None, help="Optional replay trial CSV files.")
    parser.add_argument("--trial-dir", default="artifacts/ship_3dof/trials")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--generate-trials-if-missing", action="store_true", default=False)
    parser.add_argument("--skip-residual", action="store_true", default=False)
    parser.add_argument("--log-folder", default="logs")
    parser.add_argument("--algo", default="sac")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase-steps", nargs=3, type=int, default=[600_000, 800_000, 1_100_000])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--learning-starts", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--net-arch", type=str, default="512,512,512")
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--shadow-episodes", type=int, default=50)
    parser.add_argument("--acceptance", default="scripts/ship_3dof/sim2real_acceptance.json")
    parser.add_argument("--run-tag", default=None, help="Run identifier. Used to isolate outputs.")
    parser.add_argument("--output-dir", default="artifacts/ship_3dof/runs", help="Root directory for run artifacts.")
    parser.add_argument("--report-output", default=None, help="Optional explicit pipeline report output path.")
    return parser.parse_args()


def _run(cmd: list[str]) -> float:
    print("Running:", " ".join(cmd))
    start = time.perf_counter()
    subprocess.run(cmd, check=True)
    return time.perf_counter() - start


def _default_run_tag(seed: int) -> str:
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"ship3dof_seed{seed}_{ts}"


def _resource_snapshot() -> dict[str, object]:
    snapshot: dict[str, object] = {
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }
    gpu_cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(gpu_cmd, check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError, PermissionError, OSError):
        snapshot["gpus"] = []
        snapshot["gpu_query_available"] = False
        return snapshot

    gpus = []
    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        gpus.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "utilization_gpu_pct": float(parts[2]),
                "memory_total_mb": float(parts[3]),
                "memory_used_mb": float(parts[4]),
                "memory_free_mb": float(parts[5]),
            }
        )
    snapshot["gpus"] = gpus
    snapshot["gpu_query_available"] = True
    return snapshot


def _compute_balanced_score(
    baseline_report: dict[str, object],
    residual_gate: dict[str, object],
    eval_report: dict[str, object],
    shadow_report: dict[str, object],
) -> dict[str, object]:
    m1_pass = bool(baseline_report["acceptance"]["m1_passed"])
    m2_pass = bool(residual_gate["m2_passed"])
    m3_pass = bool(eval_report["acceptance"]["m3_passed"])
    m4_pass = bool(shadow_report["acceptance"]["m4_passed"])

    success_rate = float(eval_report.get("success_rate", 0.0))
    collision_rate = float(eval_report.get("collision_rate", 1.0))
    out_of_channel_rate = float(eval_report.get("out_of_channel_rate", 1.0))
    takeover_rate = float(shadow_report.get("takeover_recommendation_rate", 1.0))
    shadow_collision_rate = float(shadow_report.get("shadow_mode_collision_rate", 1.0))

    score = 0.0
    score += 160.0 if m3_pass else -70.0
    score += 120.0 if m4_pass else -40.0
    score += 45.0 if m1_pass else -35.0
    score += 15.0 if m2_pass else -15.0

    score += 120.0 * success_rate
    score -= 100.0 * collision_rate
    score -= 55.0 * out_of_channel_rate
    score -= 80.0 * takeover_rate
    score -= 100.0 * shadow_collision_rate

    reasons = []
    if not m1_pass:
        reasons.append("M1 failed: MMG replay mismatch still high")
    if not m2_pass:
        reasons.append("M2 failed: residual model improvement below threshold")
    if not m3_pass:
        reasons.append("M3 failed: stage2 policy not robust enough")
    if not m4_pass:
        reasons.append("M4 failed: shadow takeover recommendation rate too high")

    return {
        "objective": "balanced_m3_m4",
        "score": score,
        "passes": {"M1": m1_pass, "M2": m2_pass, "M3": m3_pass, "M4": m4_pass},
        "failure_reasons": reasons,
    }


def _resolve_trials(args: argparse.Namespace) -> list[str]:
    if args.trials:
        return args.trials
    trial_dir = Path(args.trial_dir)
    trials = sorted(str(path) for path in trial_dir.glob("*.csv"))
    if trials:
        return trials
    if not args.generate_trials_if_missing:
        raise ValueError(
            "No trials provided/found. Pass --trials ... or set --generate-trials-if-missing to bootstrap synthetic data."
        )
    _run(
        [
            args.python,
            "scripts/ship_3dof/generate_mmg_trials.py",
            "--mmg-params",
            args.mmg_params,
            "--output-dir",
            args.trial_dir,
            "--dt",
            str(args.dt),
        ]
    )
    generated = sorted(str(path) for path in trial_dir.glob("*.csv"))
    if not generated:
        raise RuntimeError("Synthetic trial generation did not produce any CSV files.")
    return generated


def _latest_model(log_folder: str, algo: str) -> Path:
    run_root = Path(log_folder) / algo
    runs = sorted(run_root.glob("ShipPathTracking3DOF-v0_*"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise RuntimeError(f"No training run found under {run_root}")
    model_path = runs[-1] / "ShipPathTracking3DOF-v0.zip"
    if not model_path.exists():
        raise RuntimeError(f"Model not found: {model_path}")
    return model_path


def main() -> None:
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    started_perf = time.perf_counter()
    args = parse_args()
    run_tag = args.run_tag or _default_run_tag(args.seed)
    run_output_dir = Path(args.output_dir) / run_tag
    run_output_dir.mkdir(parents=True, exist_ok=True)
    effective_log_folder = str(Path(args.log_folder) / run_tag)

    acceptance = json.loads(Path(args.acceptance).read_text())
    trials = _resolve_trials(args)
    stage_elapsed_s: dict[str, float] = {}

    baseline_report_path = run_output_dir / "mmg_baseline_report.json"
    stage_elapsed_s["baseline"] = _run(
        [
            args.python,
            "scripts/ship_3dof/check_mmg_baseline.py",
            "--mmg-params",
            args.mmg_params,
            "--trials",
            *trials,
            "--dt",
            str(args.dt),
            "--acceptance",
            args.acceptance,
            "--output",
            str(baseline_report_path),
        ]
    )
    baseline_report = json.loads(baseline_report_path.read_text())

    residual_model_path: str | None = None
    residual_gate = {"m2_passed": True, "enabled": False, "reason": "residual skipped"}
    if not args.skip_residual:
        residual_model_path = str(run_output_dir / "residual_model.npz")
        residual_metrics_path = run_output_dir / "residual_metrics.json"
        stage_elapsed_s["residual"] = _run(
            [
                args.python,
                "scripts/ship_3dof/train_residual.py",
                "--trials",
                *trials,
                "--mmg-params",
                args.mmg_params,
                "--dt",
                str(args.dt),
                "--output",
                residual_model_path,
                "--metrics-output",
                str(residual_metrics_path),
            ]
        )
        residual_metrics = json.loads(residual_metrics_path.read_text())
        min_reduction = float(acceptance["M2"]["criteria"]["validation_loss_reduction_min"])
        reduction = float(residual_metrics["validation_loss_reduction"])
        m2_passed = reduction >= min_reduction
        residual_gate = {
            "m2_validation_loss_reduction_min": min_reduction,
            "validation_loss_reduction": reduction,
            "m2_passed": m2_passed,
            "enabled": m2_passed,
        }
        if not m2_passed:
            residual_model_path = None

    train_cmd = [
        args.python,
        "scripts/ship_3dof/train_ship_3dof.py",
        "--python",
        args.python,
        "--log-folder",
        effective_log_folder,
        "--algo",
        args.algo,
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--learning-starts",
        str(args.learning_starts),
        "--learning-rate",
        str(args.learning_rate),
        "--train-freq",
        str(args.train_freq),
        "--gradient-steps",
        str(args.gradient_steps),
        "--batch-size",
        str(args.batch_size),
        "--net-arch",
        args.net_arch,
        "--phase-steps",
        str(args.phase_steps[0]),
        str(args.phase_steps[1]),
        str(args.phase_steps[2]),
        "--mmg-params",
        args.mmg_params,
    ]
    if residual_model_path is not None:
        train_cmd += ["--residual-model", residual_model_path]
    stage_elapsed_s["training"] = _run(train_cmd)
    model_path = _latest_model(effective_log_folder, args.algo)

    eval_report_path = run_output_dir / "eval_metrics.json"
    eval_cmd = [
        args.python,
        "scripts/ship_3dof/evaluate_ship_3dof.py",
        "--model",
        str(model_path),
        "--episodes",
        str(args.eval_episodes),
        "--seed",
        str(args.seed),
        "--curriculum-stage",
        "2",
        "--acceptance",
        args.acceptance,
        "--output",
        str(eval_report_path),
        "--mmg-params",
        args.mmg_params,
    ]
    if residual_model_path is not None:
        eval_cmd += ["--residual-model", residual_model_path]
    stage_elapsed_s["evaluation"] = _run(eval_cmd)
    eval_report = json.loads(eval_report_path.read_text())

    shadow_report_path = run_output_dir / "shadow_mode_report.json"
    shadow_cmd = [
        args.python,
        "scripts/ship_3dof/run_shadow_mode.py",
        "--model",
        str(model_path),
        "--episodes",
        str(args.shadow_episodes),
        "--seed",
        str(args.seed),
        "--acceptance",
        args.acceptance,
        "--output",
        str(shadow_report_path),
        "--mmg-params",
        args.mmg_params,
    ]
    if residual_model_path is not None:
        shadow_cmd += ["--residual-model", residual_model_path]
    stage_elapsed_s["shadow"] = _run(shadow_cmd)
    shadow_report = json.loads(shadow_report_path.read_text())
    balanced = _compute_balanced_score(baseline_report, residual_gate, eval_report, shadow_report)

    final_report = {
        "run_tag": run_tag,
        "timing": {
            "started_at_utc": started_at,
            "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "elapsed_s": time.perf_counter() - started_perf,
            "stage_elapsed_s": stage_elapsed_s,
        },
        "resources": _resource_snapshot(),
        "inputs": {
            "mmg_params": args.mmg_params,
            "trial_count": len(trials),
            "residual_attempted": not args.skip_residual,
            "residual_enabled": residual_model_path is not None,
            "phase_steps": args.phase_steps,
            "learning_starts": args.learning_starts,
            "learning_rate": args.learning_rate,
            "train_freq": args.train_freq,
            "gradient_steps": args.gradient_steps,
            "batch_size": args.batch_size,
            "net_arch": args.net_arch,
            "eval_episodes": args.eval_episodes,
            "shadow_episodes": args.shadow_episodes,
            "log_folder": effective_log_folder,
            "output_dir": str(run_output_dir),
            "seed": args.seed,
        },
        "artifacts": {
            "baseline_report": str(baseline_report_path),
            "residual_model": residual_model_path,
            "model_path": str(model_path),
            "eval_report": str(eval_report_path),
            "shadow_report": str(shadow_report_path),
        },
        "gates": {
            "M1": baseline_report["acceptance"],
            "M2": residual_gate,
            "M3": eval_report["acceptance"],
            "M4": shadow_report["acceptance"],
        },
        "balanced_score": balanced,
    }
    final_report["go_for_limited_takeover"] = bool(
        final_report["gates"]["M1"]["m1_passed"]
        and final_report["gates"]["M3"]["m3_passed"]
        and final_report["gates"]["M4"]["m4_passed"]
    )
    output = Path(args.report_output) if args.report_output is not None else (run_output_dir / "pipeline_report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(final_report, indent=2))
    print(json.dumps(final_report, indent=2))


if __name__ == "__main__":
    main()
