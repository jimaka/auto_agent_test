from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ship_3dof.optimization_report import write_final_report, write_iteration_report


@dataclass(frozen=True)
class ParamCandidate:
    learning_starts: int
    learning_rate: float
    train_freq: int
    gradient_steps: int

    def key(self) -> tuple[int, float, int, int]:
        return (self.learning_starts, self.learning_rate, self.train_freq, self.gradient_steps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autotune Ship3DOF policy for a fixed wall-clock budget.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--pipeline-script", default="scripts/ship_3dof/run_known_mmg_pipeline.py")
    parser.add_argument("--mmg-params", required=True)
    parser.add_argument("--trials", nargs="+", default=None)
    parser.add_argument("--trial-dir", default="artifacts/ship_3dof/trials")
    parser.add_argument("--generate-trials-if-missing", action="store_true", default=False)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--hours", type=float, default=3.0)
    parser.add_argument("--max-runs", type=int, default=0, help="0 means unlimited within time budget.")
    parser.add_argument("--max-parallel", type=int, default=0, help="0 means auto.")
    parser.add_argument("--jobs-per-gpu", type=int, default=2, help="Used only when --max-parallel=0.")
    parser.add_argument("--output-root", default="artifacts/ship_3dof/autotune")
    parser.add_argument("--log-root", default="logs_autotune")
    parser.add_argument("--algo", default="sac")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--phase-steps", nargs=3, type=int, default=[80_000, 100_000, 140_000])
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--net-arch", type=str, default="512,512,512")
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--shadow-episodes", type=int, default=50)
    parser.add_argument("--acceptance", default="scripts/ship_3dof/sim2real_acceptance.json")
    parser.add_argument("--skip-residual", action="store_true", default=False)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--seed-offsets", nargs="+", type=int, default=[0, 13, 29])
    parser.add_argument("--learning-starts-space", nargs="+", type=int, default=[2000, 5000, 10000])
    parser.add_argument("--learning-rate-space", nargs="+", type=float, default=[1e-4, 2e-4, 3e-4])
    parser.add_argument("--train-freq-space", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--gradient-steps-space", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--elite-count", type=int, default=6)
    parser.add_argument("--report-every-batch", action="store_true", default=True)
    return parser.parse_args()


def _query_gpus() -> list[int]:
    cmd = ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError, PermissionError, OSError):
        return []
    return [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]


def _gpu_snapshot() -> list[dict[str, Any]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError, PermissionError, OSError):
        return []
    snapshots = []
    for line in result.stdout.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) != 5:
            continue
        snapshots.append(
            {
                "index": int(parts[0]),
                "utilization_gpu_pct": float(parts[1]),
                "memory_total_mb": float(parts[2]),
                "memory_used_mb": float(parts[3]),
                "memory_free_mb": float(parts[4]),
            }
        )
    return snapshots


def _choose_parallelism(args: argparse.Namespace, gpu_ids: list[int]) -> int:
    if args.max_parallel > 0:
        return args.max_parallel
    if gpu_ids:
        return max(1, len(gpu_ids) * max(1, args.jobs_per_gpu))
    return 2


def _candidate_grid(args: argparse.Namespace, rng: random.Random) -> list[ParamCandidate]:
    combos = [
        ParamCandidate(ls, lr, tf, gs)
        for ls, lr, tf, gs in itertools.product(
            args.learning_starts_space,
            args.learning_rate_space,
            args.train_freq_space,
            args.gradient_steps_space,
        )
    ]
    rng.shuffle(combos)
    return combos


def _pipeline_cmd(
    args: argparse.Namespace,
    candidate: ParamCandidate,
    run_tag: str,
    seed: int,
    output_root: Path,
    log_root: Path,
) -> list[str]:
    cmd = [
        args.python,
        args.pipeline_script,
        "--python",
        args.python,
        "--mmg-params",
        args.mmg_params,
        "--run-tag",
        run_tag,
        "--output-dir",
        str(output_root / "runs"),
        "--log-folder",
        str(log_root),
        "--algo",
        args.algo,
        "--seed",
        str(seed),
        "--device",
        args.device,
        "--phase-steps",
        str(args.phase_steps[0]),
        str(args.phase_steps[1]),
        str(args.phase_steps[2]),
        "--net-arch",
        args.net_arch,
        "--batch-size",
        str(args.batch_size),
        "--learning-starts",
        str(candidate.learning_starts),
        "--learning-rate",
        str(candidate.learning_rate),
        "--train-freq",
        str(candidate.train_freq),
        "--gradient-steps",
        str(candidate.gradient_steps),
        "--eval-episodes",
        str(args.eval_episodes),
        "--shadow-episodes",
        str(args.shadow_episodes),
        "--acceptance",
        args.acceptance,
        "--dt",
        str(args.dt),
    ]
    if args.skip_residual:
        cmd.append("--skip-residual")
    if args.trials:
        cmd.extend(["--trials", *args.trials])
    else:
        cmd.extend(["--trial-dir", args.trial_dir])
        if args.generate_trials_if_missing:
            cmd.append("--generate-trials-if-missing")
    return cmd


def _run_candidate(
    args: argparse.Namespace,
    candidate: ParamCandidate,
    seed: int,
    run_tag: str,
    worker_idx: int,
    gpu_id: int | None,
    output_root: Path,
    log_root: Path,
) -> dict[str, Any]:
    run_logs_dir = output_root / "run_logs"
    run_logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_logs_dir / f"{run_tag}.log"
    cmd = _pipeline_cmd(args, candidate, run_tag, seed, output_root, log_root)
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    started = time.perf_counter()
    with log_path.open("w") as f:
        process = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, text=True)
    elapsed = time.perf_counter() - started
    report_path = output_root / "runs" / run_tag / "pipeline_report.json"

    payload: dict[str, Any] = {
        "run_tag": run_tag,
        "worker_idx": worker_idx,
        "gpu_id": gpu_id,
        "candidate": {
            "learning_starts": candidate.learning_starts,
            "learning_rate": candidate.learning_rate,
            "train_freq": candidate.train_freq,
            "gradient_steps": candidate.gradient_steps,
            "seed": seed,
        },
        "elapsed_s": elapsed,
        "return_code": process.returncode,
        "log_path": str(log_path),
        "report_path": str(report_path),
        "status": "failed",
        "score": -1e9,
        "gates": {},
        "metrics": {},
    }
    if process.returncode == 0 and report_path.exists():
        report = json.loads(report_path.read_text())
        eval_metrics: dict[str, Any] = {}
        shadow_metrics: dict[str, Any] = {}
        eval_path = report.get("artifacts", {}).get("eval_report")
        shadow_path = report.get("artifacts", {}).get("shadow_report")
        if isinstance(eval_path, str) and Path(eval_path).exists():
            eval_metrics = json.loads(Path(eval_path).read_text())
        if isinstance(shadow_path, str) and Path(shadow_path).exists():
            shadow_metrics = json.loads(Path(shadow_path).read_text())
        payload["status"] = "completed"
        payload["score"] = float(report.get("balanced_score", {}).get("score", -1e9))
        payload["gates"] = report.get("gates", {})
        payload["metrics"] = {
            "success_rate": float(eval_metrics.get("success_rate", 0.0)),
            "collision_rate": float(eval_metrics.get("collision_rate", 1.0)),
            "out_of_channel_rate": float(eval_metrics.get("out_of_channel_rate", 1.0)),
            "takeover_recommendation_rate": float(shadow_metrics.get("takeover_recommendation_rate", 1.0)),
        }
    return payload


def _candidate_summary(records: list[dict[str, Any]]) -> dict[tuple[int, float, int, int], float]:
    grouped: dict[tuple[int, float, int, int], list[float]] = {}
    for item in records:
        if item.get("status") != "completed":
            continue
        params = item.get("candidate", {})
        key = (
            int(params.get("learning_starts", 0)),
            float(params.get("learning_rate", 0.0)),
            int(params.get("train_freq", 0)),
            int(params.get("gradient_steps", 0)),
        )
        grouped.setdefault(key, []).append(float(item.get("score", -1e9)))
    return {key: sum(scores) / len(scores) for key, scores in grouped.items()}


def main() -> None:
    args = parse_args()
    started_perf = time.perf_counter()
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()

    output_root = Path(args.output_root)
    reports_dir = output_root / "optimization_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    log_root = Path(args.log_root)
    log_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.base_seed)
    gpu_ids = _query_gpus()
    parallelism = _choose_parallelism(args, gpu_ids)

    candidates = _candidate_grid(args, rng)
    pending: list[tuple[ParamCandidate, int]] = []
    seen_runs: set[tuple[tuple[int, float, int, int], int]] = set()
    seed_offsets = args.seed_offsets
    for cand in candidates:
        seed = args.base_seed + seed_offsets[0]
        pending.append((cand, seed))
        seen_runs.add((cand.key(), seed))

    completed: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    run_count = 0
    iteration_idx = 0
    budget_s = max(0.0, args.hours * 3600.0)

    while pending and (time.perf_counter() - started_perf) < budget_s:
        if args.max_runs > 0 and run_count >= args.max_runs:
            break
        iteration_idx += 1
        batch: list[tuple[ParamCandidate, int]] = []
        while pending and len(batch) < parallelism:
            if args.max_runs > 0 and (run_count + len(batch)) >= args.max_runs:
                break
            batch.append(pending.pop(0))
        if not batch:
            break

        batch_started = time.perf_counter()
        batch_results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=len(batch)) as pool:
            futures = []
            for worker_idx, (candidate, seed) in enumerate(batch):
                gpu_id = gpu_ids[worker_idx % len(gpu_ids)] if gpu_ids else None
                run_tag = (
                    f"iter{iteration_idx:03d}_w{worker_idx}_"
                    f"ls{candidate.learning_starts}_lr{candidate.learning_rate:.1e}_"
                    f"tf{candidate.train_freq}_gs{candidate.gradient_steps}_s{seed}"
                )
                futures.append(
                    pool.submit(
                        _run_candidate,
                        args,
                        candidate,
                        seed,
                        run_tag,
                        worker_idx,
                        gpu_id,
                        output_root,
                        log_root,
                    )
                )
            for fut in as_completed(futures):
                item = fut.result()
                batch_results.append(item)
                completed.append(item)
                run_count += 1
                if item.get("status") == "completed":
                    if best is None or float(item.get("score", -1e9)) > float(best.get("score", -1e9)):
                        best = item

        # Elimination + exploitation: keep only elite parameter groups for additional seeds.
        summary = _candidate_summary(completed)
        elite_keys = {
            key for key, _ in sorted(summary.items(), key=lambda pair: pair[1], reverse=True)[: max(1, args.elite_count)]
        }
        for key in elite_keys:
            candidate = ParamCandidate(
                learning_starts=key[0],
                learning_rate=key[1],
                train_freq=key[2],
                gradient_steps=key[3],
            )
            for offset in seed_offsets[1:]:
                seed = args.base_seed + offset
                run_key = (candidate.key(), seed)
                if run_key in seen_runs:
                    continue
                pending.append((candidate, seed))
                seen_runs.add(run_key)

        batch_elapsed = time.perf_counter() - batch_started
        report_path = reports_dir / f"iteration_{iteration_idx:03d}.md"
        write_iteration_report(
            output_path=report_path,
            iteration_idx=iteration_idx,
            run_results=batch_results,
            best_record=best,
            elapsed_s=batch_elapsed,
        )

        iteration_json = {
            "iteration": iteration_idx,
            "batch_elapsed_s": batch_elapsed,
            "batch_size": len(batch_results),
            "gpu_snapshot": _gpu_snapshot(),
            "best_run_tag": best.get("run_tag") if best else None,
            "best_score": float(best.get("score", -1e9)) if best else None,
            "batch_results": batch_results,
        }
        (reports_dir / f"iteration_{iteration_idx:03d}.json").write_text(json.dumps(iteration_json, indent=2))

    elapsed_s = time.perf_counter() - started_perf
    final_summary = {
        "started_at_utc": started_at,
        "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "time_budget_hours": args.hours,
        "elapsed_s": elapsed_s,
        "parallelism": parallelism,
        "gpu_ids": gpu_ids,
        "total_runs": run_count,
        "best_run": best,
        "completed_runs": completed,
        "reports_dir": str(reports_dir),
    }
    summary_path = output_root / "autotune_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(final_summary, indent=2))
    write_final_report(output_root / "autotune_summary.md", final_summary)
    print(json.dumps(final_summary, indent=2))


if __name__ == "__main__":
    main()
