from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch independent Ship3DOF training jobs across multiple GPUs."
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--mode", choices=["sac", "pipeline"], default="sac")
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU indices. Auto-detect when omitted.")
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument("--max-parallel", type=int, default=0, help="0 means auto from GPUs x jobs-per-gpu.")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--output-root", default="artifacts/ship_3dof/multi_gpu")
    parser.add_argument("--log-root", default="logs_multi_gpu")
    parser.add_argument("--mmg-params", default="scripts/ship_3dof/mmg_params_example.json")
    parser.add_argument("--trial-dir", default="artifacts/ship_3dof/trials")
    parser.add_argument("--generate-trials-if-missing", action="store_true", default=False)
    parser.add_argument("--skip-residual", action="store_true", default=False)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--phase-steps", nargs=3, type=int, default=[600_000, 800_000, 1_100_000])
    parser.add_argument("--learning-starts", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--net-arch", type=str, default="512,512,512")
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--shadow-episodes", type=int, default=50)
    parser.add_argument("--acceptance", default="scripts/ship_3dof/sim2real_acceptance.json")
    parser.add_argument("--total-override", type=int, default=None, help="SAC mode only: single-phase step count.")
    parser.add_argument("--smoke", action="store_true", default=False, help="Use short-step smoke preset.")
    return parser.parse_args()


def _query_gpus() -> list[int]:
    cmd = ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError, PermissionError, OSError):
        return []
    return [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]


def _parse_gpu_list(value: str | None) -> list[int]:
    if value is None or not value.strip():
        return _query_gpus()
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _choose_parallelism(args: argparse.Namespace, gpu_ids: list[int]) -> int:
    if args.max_parallel > 0:
        return args.max_parallel
    if gpu_ids:
        return max(1, len(gpu_ids) * max(1, args.jobs_per_gpu))
    return 1


def _resolve_seeds(args: argparse.Namespace, job_count: int) -> list[int]:
    if args.seeds:
        return list(args.seeds)
    return [args.base_seed + idx for idx in range(job_count)]


def _apply_smoke_preset(args: argparse.Namespace) -> None:
    args.phase_steps = [20_000, 20_000, 30_000]
    args.learning_starts = 2000
    args.batch_size = 1024
    args.net_arch = "256,256"
    args.eval_episodes = 10
    args.shadow_episodes = 10
    args.generate_trials_if_missing = True


def _run_tag(gpu_id: int | None, seed: int, worker_idx: int) -> str:
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
    gpu_part = f"g{gpu_id}" if gpu_id is not None else "cpu"
    return f"multi_gpu_{gpu_part}_s{seed}_w{worker_idx}_{ts}"


def _sac_cmd(args: argparse.Namespace, seed: int, log_folder: str) -> list[str]:
    cmd = [
        args.python,
        str(SCRIPTS_DIR / "train_ship_3dof.py"),
        "--python",
        args.python,
        "--log-folder",
        log_folder,
        "--seed",
        str(seed),
        "--device",
        args.device,
        "--mmg-params",
        args.mmg_params,
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
    ]
    if args.total_override is not None:
        cmd += ["--total-override", str(args.total_override)]
    else:
        cmd += [
            "--phase-steps",
            str(args.phase_steps[0]),
            str(args.phase_steps[1]),
            str(args.phase_steps[2]),
        ]
    return cmd


def _pipeline_cmd(
    args: argparse.Namespace,
    seed: int,
    run_tag: str,
    log_folder: str,
    output_dir: Path,
) -> list[str]:
    cmd = [
        args.python,
        str(SCRIPTS_DIR / "run_known_mmg_pipeline.py"),
        "--python",
        args.python,
        "--mmg-params",
        args.mmg_params,
        "--trial-dir",
        args.trial_dir,
        "--log-folder",
        log_folder,
        "--output-dir",
        str(output_dir),
        "--run-tag",
        run_tag,
        "--seed",
        str(seed),
        "--device",
        args.device,
        "--phase-steps",
        str(args.phase_steps[0]),
        str(args.phase_steps[1]),
        str(args.phase_steps[2]),
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
        "--eval-episodes",
        str(args.eval_episodes),
        "--shadow-episodes",
        str(args.shadow_episodes),
        "--acceptance",
        args.acceptance,
    ]
    if args.generate_trials_if_missing:
        cmd.append("--generate-trials-if-missing")
    if args.skip_residual:
        cmd.append("--skip-residual")
    return cmd


def _latest_model(log_folder: str) -> Path | None:
    run_root = Path(log_folder) / "sac"
    runs = sorted(run_root.glob("ShipPathTracking3DOF-v0_*"), key=lambda p: p.stat().st_mtime)
    if not runs:
        return None
    model_path = runs[-1] / "ShipPathTracking3DOF-v0.zip"
    return model_path if model_path.exists() else None


def _run_job(
    args: argparse.Namespace,
    seed: int,
    worker_idx: int,
    gpu_id: int | None,
    output_root: Path,
    log_root: Path,
) -> dict[str, Any]:
    run_tag = _run_tag(gpu_id, seed, worker_idx)
    job_log_folder = str(log_root / run_tag)
    run_logs_dir = output_root / "run_logs"
    run_logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_logs_dir / f"{run_tag}.log"

    if args.mode == "sac":
        cmd = _sac_cmd(args, seed, job_log_folder)
        report_path: Path | None = None
    else:
        pipeline_output_dir = output_root / "runs"
        cmd = _pipeline_cmd(args, seed, run_tag, job_log_folder, pipeline_output_dir)
        report_path = pipeline_output_dir / run_tag / "pipeline_report.json"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    started = time.perf_counter()
    with log_path.open("w") as log_file:
        process = subprocess.run(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            cwd=PROJECT_ROOT,
        )
    elapsed = time.perf_counter() - started

    model_path = _latest_model(job_log_folder)
    payload: dict[str, Any] = {
        "run_tag": run_tag,
        "mode": args.mode,
        "worker_idx": worker_idx,
        "gpu_id": gpu_id,
        "seed": seed,
        "elapsed_s": elapsed,
        "return_code": process.returncode,
        "log_path": str(log_path),
        "log_folder": job_log_folder,
        "model_path": str(model_path) if model_path is not None else None,
        "report_path": str(report_path) if report_path is not None else None,
        "status": "completed" if process.returncode == 0 else "failed",
    }

    if process.returncode == 0 and report_path is not None and report_path.exists():
        report = json.loads(report_path.read_text())
        payload["balanced_score"] = float(report.get("balanced_score", {}).get("score", -1e9))
        payload["gates"] = report.get("gates", {})
        payload["go_for_limited_takeover"] = bool(report.get("go_for_limited_takeover", False))

    return payload


def main() -> None:
    args = parse_args()
    if args.smoke:
        _apply_smoke_preset(args)

    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    started_perf = time.perf_counter()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    log_root = Path(args.log_root)
    log_root.mkdir(parents=True, exist_ok=True)

    gpu_ids = _parse_gpu_list(args.gpus)
    parallelism = _choose_parallelism(args, gpu_ids)
    seeds = _resolve_seeds(args, parallelism)
    if len(seeds) < parallelism:
        seeds.extend(args.base_seed + len(seeds) + idx for idx in range(parallelism - len(seeds)))
    jobs = list(enumerate(seeds[:parallelism]))

    print(f"Multi-GPU training: mode={args.mode}, gpus={gpu_ids}, parallelism={parallelism}")

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = []
        for worker_idx, seed in jobs:
            gpu_id = gpu_ids[worker_idx % len(gpu_ids)] if gpu_ids else None
            futures.append(
                pool.submit(_run_job, args, seed, worker_idx, gpu_id, output_root, log_root)
            )
        for fut in as_completed(futures):
            result = fut.result()
            results.append(result)
            print(f"  [{result['status']}] gpu={result['gpu_id']} seed={result['seed']} tag={result['run_tag']}")

    results.sort(key=lambda item: int(item["worker_idx"]))
    summary = {
        "started_at_utc": started_at,
        "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "elapsed_s": time.perf_counter() - started_perf,
        "mode": args.mode,
        "parallelism": parallelism,
        "gpu_ids": gpu_ids,
        "jobs_per_gpu": args.jobs_per_gpu,
        "smoke": args.smoke,
        "jobs": results,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
