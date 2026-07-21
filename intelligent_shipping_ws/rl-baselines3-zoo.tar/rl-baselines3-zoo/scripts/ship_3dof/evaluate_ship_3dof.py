from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import custom_envs  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ShipPathTracking3DOF-v0 policy.")
    parser.add_argument("--model", required=True, help="Path to SAC model zip.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", default="artifacts/ship_3dof/eval_metrics.json")
    parser.add_argument("--curriculum-stage", type=int, default=2)
    parser.add_argument("--mmg-params", type=str, default=None, help="JSON file with MMG parameters.")
    parser.add_argument("--residual-model", type=str, default=None, help="Optional residual model .npz path.")
    parser.add_argument(
        "--acceptance",
        default="scripts/ship_3dof/sim2real_acceptance.json",
        help="Acceptance config used to evaluate pass/fail.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_kwargs: dict[str, object] = {"curriculum_stage": args.curriculum_stage}
    if args.mmg_params is not None:
        env_kwargs["mmg_params_path"] = args.mmg_params
    if args.residual_model is not None:
        env_kwargs["residual_model_path"] = args.residual_model
    env = gym.make("ShipPathTracking3DOF-v0", **env_kwargs)
    model = SAC.load(args.model)
    rng = np.random.default_rng(args.seed)

    returns = []
    success = 0
    collisions = 0
    out_of_channel = 0
    for _ in range(args.episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**32 - 1)))
        done = False
        truncated = False
        episode_return = 0.0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_return += reward
        returns.append(episode_return)
        if info.get("success", False):
            success += 1
        if info.get("collision", False):
            collisions += 1
        if info.get("out_of_channel", False):
            out_of_channel += 1

    metrics = {
        "episodes": args.episodes,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "success_rate": float(success / args.episodes),
        "collision_rate": float(collisions / args.episodes),
        "out_of_channel_rate": float(out_of_channel / args.episodes),
    }
    acceptance_payload = json.loads(Path(args.acceptance).read_text())
    m3_criteria = acceptance_payload["M3"]["criteria"]
    metrics["acceptance"] = {
        "m3_success_rate_min": float(m3_criteria["success_rate_min"]),
        "m3_collision_rate_max": float(m3_criteria["collision_rate_max"]),
        "m3_passed": bool(
            metrics["success_rate"] >= float(m3_criteria["success_rate_min"])
            and metrics["collision_rate"] <= float(m3_criteria["collision_rate_max"])
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
