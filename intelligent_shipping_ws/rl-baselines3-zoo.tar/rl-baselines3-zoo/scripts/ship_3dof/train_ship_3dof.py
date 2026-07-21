from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAC curriculum training for ShipPathTracking3DOF-v0.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--log-folder", default="logs")
    parser.add_argument("--algo", default="sac")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase-steps", nargs=3, type=int, default=[600_000, 800_000, 1_100_000])
    parser.add_argument("--total-override", type=int, default=None, help="If set, train in one phase with this step count.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mmg-params", type=str, default=None, help="JSON file with MMG parameters.")
    parser.add_argument("--residual-model", type=str, default=None, help="Optional residual model .npz path.")
    parser.add_argument("--learning-starts", type=int, default=5000, help="Override learning_starts for SAC.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Override learning rate.")
    parser.add_argument("--train-freq", type=int, default=4, help="Override train_freq.")
    parser.add_argument("--gradient-steps", type=int, default=4, help="Override gradient_steps.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Override batch_size. 1024 为本环境实测最优点（更新延迟最低且样本效率更高）；"
        "降更新节奏（如 2048x1）虽 fps 高 2 倍但学习质量显著恶化，勿用。",
    )
    parser.add_argument(
        "--no-accel",
        action="store_true",
        default=False,
        help="禁用 AccelerateCallback（TF32）。默认开启。",
    )
    parser.add_argument(
        "--net-arch",
        type=str,
        default="512,512,512",
        help="Comma-separated hidden sizes for policy network (e.g. 256,256 or 512,512,512).",
    )
    return parser.parse_args()


def _parse_net_arch(value: str) -> list[int]:
    parts = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
    if not parts:
        raise ValueError("--net-arch must contain at least one layer size.")
    arch = [int(part) for part in parts]
    if any(layer <= 0 for layer in arch):
        raise ValueError("--net-arch values must be positive integers.")
    return arch


def _latest_run(base: Path, env_id: str) -> Path | None:
    if not base.exists():
        return None
    runs = sorted(base.glob(f"{env_id}_*"), key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def _run_train(
    args: argparse.Namespace,
    n_steps: int,
    stage: int | None,
    trained_agent: str | None,
) -> Path:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer.")
    net_arch = _parse_net_arch(args.net_arch)
    cmd = [
        args.python,
        "train.py",
        "--algo",
        args.algo,
        "--env",
        "ShipPathTracking3DOF-v0",
        "--log-folder",
        args.log_folder,
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--n-timesteps",
        str(n_steps),
    ]
    env_kwargs = []
    if stage is not None:
        env_kwargs.append(f"curriculum_stage:{stage}")
    if args.mmg_params is not None:
        env_kwargs.append(f"mmg_params_path:'{args.mmg_params}'")
    if args.residual_model is not None:
        env_kwargs.append(f"residual_model_path:'{args.residual_model}'")
    if env_kwargs:
        cmd += ["--env-kwargs", *env_kwargs]
    hyperparams = [
        f"learning_starts:{args.learning_starts}",
        f"learning_rate:{args.learning_rate}",
        f"train_freq:{args.train_freq}",
        f"gradient_steps:{args.gradient_steps}",
        f"batch_size:{args.batch_size}",
        f"policy_kwargs:dict(net_arch={net_arch})",
    ]
    if not args.no_accel:
        # 字符串值需内嵌单引号（rl_zoo3 StoreDict 用 eval 解析）
        hyperparams.append("callback:'custom_envs.ship_3dof.accel.AccelerateCallback'")
    cmd += ["--hyperparams", *hyperparams]
    if trained_agent is not None:
        cmd += ["--trained-agent", trained_agent]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    run_root = Path(args.log_folder) / args.algo
    latest = _latest_run(run_root, "ShipPathTracking3DOF-v0")
    if latest is None:
        raise RuntimeError("Training completed but no run folder found.")
    model_path = latest / "ShipPathTracking3DOF-v0.zip"
    if not model_path.exists():
        raise RuntimeError(f"Model not found: {model_path}")
    return model_path


def _summarize_monitor(log_folder: str) -> None:
    monitor_files = sorted(Path(log_folder).glob("sac/ShipPathTracking3DOF-v0_*/0.monitor.csv"))
    if not monitor_files:
        print("No monitor files found for summary.")
        return
    latest = monitor_files[-1]
    returns = []
    with latest.open() as f:
        lines = [line for line in f.readlines() if not line.startswith("#")]
    if not lines:
        print("Monitor file has no data rows.")
        return
    for row in csv.DictReader(lines):
        if "r" not in row:
            continue
        returns.append(float(row["r"]))
    if not returns:
        print("Monitor file has no episodes yet.")
        return
    tail = returns[-50:] if len(returns) >= 50 else returns
    print(f"episodes={len(returns)}")
    print(f"mean_return_last50={sum(tail) / len(tail):.3f}")
    print(f"best_return={max(returns):.3f}")


def main() -> None:
    args = parse_args()
    model_path = None

    if args.total_override is not None:
        model_path = _run_train(args, n_steps=args.total_override, stage=None, trained_agent=None)
    else:
        for stage, steps in enumerate(args.phase_steps):
            model_path = _run_train(
                args,
                n_steps=steps,
                stage=stage,
                trained_agent=str(model_path) if model_path is not None else None,
            )

    print(f"Final model: {model_path}")
    _summarize_monitor(args.log_folder)


if __name__ == "__main__":
    main()
