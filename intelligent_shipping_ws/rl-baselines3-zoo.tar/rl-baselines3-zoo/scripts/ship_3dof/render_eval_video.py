#!/usr/bin/env python3
"""将训练好的 ShipPathTracking3DOF-v0 策略回放并渲染为 MP4 视频。

环境本身不带 render 模式，本脚本读取 env.unwrapped 内部状态
（船位、航向、舵角、转速、参考路径）用 matplotlib 绘制动画。

用法:
    python scripts/ship_3dof/render_eval_video.py \
        --model artifacts/ship_3dof/runs/demo_smoke/.../ShipPathTracking3DOF-v0.zip \
        --output artifacts/demo_video.mp4 \
        --curriculum-stage 2 --seed 7
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.patches import Polygon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="SB3 模型 zip 路径")
    parser.add_argument("--output", default="artifacts/ship_3dof/demo_video.mp4")
    parser.add_argument("--curriculum-stage", type=int, default=2)
    parser.add_argument("--mmg-params", default=None)
    parser.add_argument("--residual-model", default=None)
    parser.add_argument("--vecnormalize", default=None,
                        help="VecNormalize.pkl 路径；缺省自动在模型同级及嵌套目录查找")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--stride", type=int, default=2, help="每 N 个仿真步渲染 1 帧")
    parser.add_argument("--view-len", type=float, default=140.0, help="跟随窗口长度 (m)")
    return parser.parse_args()


def make_env(args: argparse.Namespace):
    import gymnasium as gym

    import custom_envs  # noqa: F401  注册 ShipPathTracking3DOF-v0

    env_kwargs = {"curriculum_stage": args.curriculum_stage}
    if args.mmg_params:
        env_kwargs["mmg_params_path"] = args.mmg_params
    if args.residual_model and os.path.exists(args.residual_model):
        env_kwargs["residual_model_path"] = args.residual_model
    return gym.make("ShipPathTracking3DOF-v0", **env_kwargs)


def find_vecnormalize(args: argparse.Namespace) -> str | None:
    """在模型同级目录与 rl_zoo3 嵌套目录中查找 vecnormalize.pkl。"""
    if args.vecnormalize:
        return args.vecnormalize
    model_dir = os.path.dirname(os.path.abspath(args.model))
    candidates = [
        os.path.join(model_dir, "vecnormalize.pkl"),
        os.path.join(model_dir, "ShipPathTracking3DOF-v0", "vecnormalize.pkl"),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return None


def rollout(env, model, max_steps: int, seed: int, vecnormalize_path: str | None = None) -> dict:
    from custom_envs.ship_3dof.path import wrap_to_pi

    vec_env = None
    if vecnormalize_path:
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        # 训练时启用了 VecNormalize(norm_obs=True)：评估必须加载同样的归一化统计，
        # 否则策略收到未归一化观测会输出饱和动作
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize.load(vecnormalize_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        vec_env.seed(seed)
        obs = vec_env.reset()
    else:
        obs, _ = env.reset(seed=seed)

    core = env.unwrapped
    data = {k: [] for k in ("x", "y", "psi", "u", "r", "delta", "n", "reward", "e_y", "e_psi")}
    # 记录本次回合实际使用的参考路径，供渲染（VecNormalize.reset 会再次 reset 底层 env，
    # 不能依赖外部同种子重建的路径）
    data["path_x"] = core.path._x.copy()  # noqa: SLF001
    data["path_y"] = core.path._y.copy()  # noqa: SLF001
    ep_return, success, done = 0.0, False, False
    steps = 0
    while not done and steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        if vec_env is not None:
            obs, reward_arr, done_arr, infos = vec_env.step(action)
            reward = float(reward_arr[0])
            terminated = bool(done_arr[0])
            truncated = False
            info = infos[0]
        else:
            obs, reward, terminated, truncated, info = env.step(action)
        core = env.unwrapped
        st, path = core.state, core.path
        point = path.closest_point(st.x, st.y)
        e_y = path.signed_cross_track_error(st.x, st.y, point)
        data["x"].append(st.x)
        data["y"].append(st.y)
        data["psi"].append(st.psi)
        data["u"].append(st.u)
        data["r"].append(st.r)
        data["delta"].append(st.delta)
        data["n"].append(st.n)
        data["reward"].append(float(reward))
        data["e_y"].append(e_y)
        data["e_psi"].append(float(wrap_to_pi(st.psi - point.heading)))
        ep_return += float(reward)
        success = bool(info.get("success", False))
        done = terminated or truncated
        steps += 1
    if vec_env is not None:
        vec_env.close()
    else:
        env.close()
    data["ep_return"] = ep_return
    data["success"] = success
    data["steps"] = steps
    return data


def ship_triangle(x: float, y: float, psi: float, scale: float = 6.0) -> np.ndarray:
    """返回船体三角形的 4 个顶点（闭合），船头朝 psi 方向。"""
    hull = np.array([[1.0, 0.0], [-0.6, 0.45], [-0.6, -0.45], [1.0, 0.0]]) * scale
    c, s = np.cos(psi), np.sin(psi)
    rot = np.array([[c, -s], [s, c]])
    return hull @ rot.T + np.array([x, y])


def render(data: dict, args: argparse.Namespace) -> None:
    px, py = data["path_x"], data["path_y"]
    xs = np.asarray(data["x"])
    ys = np.asarray(data["y"])
    idx = np.arange(0, data["steps"], args.stride)
    if len(idx) == 0:
        raise RuntimeError("rollout 没有产生任何步，无法渲染")

    fig, (ax_map, ax_err) = plt.subplots(
        1, 2, figsize=(12.8, 6.0), gridspec_kw={"width_ratios": [2.6, 1.4]}
    )
    fig.patch.set_facecolor("white")

    # ---- 左：俯视轨迹图（跟随窗口） ----
    ax_map.plot(px, py, color="#9ca3af", lw=1.5, label="reference path")
    ax_map.plot(px, py + 8.0, color="#d1d5db", lw=0.8, ls="--")
    ax_map.plot(px, py - 8.0, color="#d1d5db", lw=0.8, ls="--", label="channel edge")
    (trail,) = ax_map.plot([], [], color="#2563eb", lw=2.0, label="ship track")
    ship_patch = Polygon(ship_triangle(xs[0], ys[0], data["psi"][0]), closed=True,
                         fc="#dc2626", ec="#7f1d1d", zorder=5)
    ax_map.add_patch(ship_patch)
    hud = ax_map.text(0.02, 0.97, "", transform=ax_map.transAxes, va="top",
                      fontsize=9, family="monospace",
                      bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#d1d5db", alpha=0.9))
    ax_map.set_xlabel("x [m]")
    ax_map.set_ylabel("y [m]")
    ax_map.legend(loc="lower right", fontsize=8)
    ax_map.set_title("ShipPathTracking3DOF-v0 evaluation rollout")

    # ---- 右：横向误差曲线 ----
    t_all = np.arange(data["steps"]) * 0.1
    ax_err.axhline(0.0, color="#9ca3af", lw=0.8)
    ax_err.axhline(8.0, color="#dc2626", lw=0.8, ls="--")
    ax_err.axhline(-8.0, color="#dc2626", lw=0.8, ls="--")
    ax_err.plot(t_all, data["e_y"], color="#f3f4f6", lw=1.0)
    (err_line,) = ax_err.plot([], [], color="#2563eb", lw=1.8)
    (err_dot,) = ax_err.plot([], [], "o", color="#dc2626", ms=5)
    ax_err.set_xlim(0, max(t_all[idx[-1]], 1.0))
    ax_err.set_ylim(-12, 12)
    ax_err.set_xlabel("t [s]")
    ax_err.set_ylabel("cross-track error $e_y$ [m]")
    ax_err.set_title("Tracking error")

    fig.tight_layout()

    # 显式指定 libx264 + yuv420p + faststart，保证常见播放器（VLC/Totem/Windows）兼容
    writer = FFMpegWriter(
        fps=args.fps,
        codec="libx264",
        bitrate=2400,
        extra_args=["-pix_fmt", "yuv420p", "-profile:v", "main", "-movflags", "+faststart"],
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    half = args.view_len / 2.0

    def update(frame_i: int):
        i = int(idx[frame_i])
        x, y, psi = xs[i], ys[i], data["psi"][i]
        trail.set_data(xs[: i + 1], ys[: i + 1])
        ship_patch.set_xy(ship_triangle(x, y, psi))
        err_line.set_data(t_all[: i + 1], data["e_y"][: i + 1])
        err_dot.set_data([t_all[i]], [data["e_y"][i]])
        cum_r = float(np.sum(data["reward"][: i + 1]))
        hud.set_text(
            f"t={t_all[i]:6.1f}s  u={data['u'][i]:4.2f}m/s  r={data['r'][i]:+.3f}\n"
            f"e_y={data['e_y'][i]:+6.2f}m  e_psi={np.rad2deg(data['e_psi'][i]):+5.1f}deg\n"
            f"delta={np.rad2deg(data['delta'][i]):+5.1f}deg  n={data['n'][i]:4.1f}rpm\n"
            f"return={cum_r:8.1f}"
        )
        # 跟随窗口：以船当前位置为中心，y 范围按局部路径居中
        ax_map.set_xlim(x - half * 0.7, x + half * 1.3)
        y_center = float(np.interp(x, px, py))
        ax_map.set_ylim(y_center - 45, y_center + 45)
        return trail, ship_patch, err_line, err_dot, hud

    with writer.saving(fig, args.output, dpi=110):
        for f in range(len(idx)):
            update(f)
            writer.grab_frame()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    from stable_baselines3 import SAC

    model = SAC.load(args.model)
    vecnormalize_path = find_vecnormalize(args)
    if vecnormalize_path:
        print(f"using VecNormalize stats: {vecnormalize_path}")
    env = make_env(args)
    data = rollout(env, model, args.max_steps, args.seed, vecnormalize_path)
    render(data, args)

    print(f"video saved: {args.output}")
    print(f"episodes steps={data['steps']} return={data['ep_return']:.1f} success={data['success']}")


if __name__ == "__main__":
    main()
