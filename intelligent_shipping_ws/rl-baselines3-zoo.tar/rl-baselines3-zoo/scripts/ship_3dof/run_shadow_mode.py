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
from custom_envs.ship_3dof.deploy import FallbackPIDController, SafetyFilter, SafetyLimits, ShadowModeSupervisor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run shadow-mode safety assessment for ship policy.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", default="artifacts/ship_3dof/shadow_mode_report.json")
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
    env_kwargs: dict[str, object] = {"curriculum_stage": 2}
    if args.mmg_params is not None:
        env_kwargs["mmg_params_path"] = args.mmg_params
    if args.residual_model is not None:
        env_kwargs["residual_model_path"] = args.residual_model
    env = gym.make("ShipPathTracking3DOF-v0", **env_kwargs)
    model = SAC.load(args.model)
    supervisor = ShadowModeSupervisor()
    fallback = FallbackPIDController()
    limits = SafetyLimits(
        delta_max=np.deg2rad(35.0),
        dot_delta_max=np.deg2rad(4.0),
        n_min=0.0,
        n_max=25.0,
        dot_n_max=2.0,
        max_abs_r=1.1,
        max_abs_e_y=8.0,
    )
    safety_filter = SafetyFilter(limits=limits, dt=0.1)

    rng = np.random.default_rng(args.seed)
    success = 0
    filtered_total = 0
    collision_count = 0
    for _ in range(args.episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**32 - 1)))
        done = False
        truncated = False
        episode_collision = False
        while not (done or truncated):
            proposed, _ = model.predict(obs, deterministic=True)
            raw = env.unwrapped  # type: ignore[attr-defined]
            e_y, e_psi, _, _ = raw._tracking_error()
            safe_action, overridden = safety_filter.filter_action(
                action=np.asarray(proposed, dtype=np.float64),
                current_delta=raw.state.delta,
                current_n=raw.state.n,
                estimated_r=raw.state.r,
                estimated_e_y=e_y,
            )
            if overridden:
                safe_action = fallback.action(
                    e_y=e_y,
                    e_psi=e_psi,
                    r=raw.state.r,
                    current_n=raw.state.n,
                    dot_delta_max=limits.dot_delta_max,
                    dot_n_max=limits.dot_n_max,
                )
                filtered_total += 1
            supervisor.observe(overridden)
            obs, _, done, truncated, info = env.step(safe_action)
            episode_collision = episode_collision or bool(info.get("collision", False))
        if info.get("success", False):
            success += 1
        if episode_collision:
            collision_count += 1

    acceptance_payload = json.loads(Path(args.acceptance).read_text())
    m4_criteria = acceptance_payload["M4"]["criteria"]
    takeover_rate = supervisor.recommendation_rate()
    report = {
        "episodes": args.episodes,
        "success_rate": success / args.episodes,
        "takeover_recommendation_rate": takeover_rate,
        "total_takeover_recommendations": filtered_total,
        "shadow_mode_collision_count": collision_count,
        "shadow_mode_collision_rate": collision_count / args.episodes,
        "acceptance": {
            "m4_takeover_rate_max": float(m4_criteria["takeover_recommendation_rate_max"]),
            "m4_shadow_collision_count_max": int(m4_criteria["shadow_mode_collision_count_max"]),
            "m4_passed": bool(
                takeover_rate <= float(m4_criteria["takeover_recommendation_rate_max"])
                and collision_count <= int(m4_criteria["shadow_mode_collision_count_max"])
            ),
        },
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
