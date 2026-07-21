#!/usr/bin/env python3
"""生成《船舶强化学习训练模型说明文档》配图。"""
import csv
import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FIGDIR = "docs/design/figures"
C_CLI = "#dbeafe"
C_CORE = "#dcfce7"
C_ALGO = "#fef3c7"
C_ENV = "#fee2e2"
C_OUT = "#f3e8ff"
C_GATE = "#ffedd5"
EDGE = "#374151"


def box(ax, x, y, w, h, text, fc, fs=10, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=1.2, edgecolor=EDGE, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight, zorder=3)


def arrow(ax, x1, y1, x2, y2, text="", fs=8.5, rad=0.0, color=EDGE, toff=(0, 0.12)):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                 linewidth=1.3, color=color, connectionstyle=f"arc3,rad={rad}", zorder=1))
    if text:
        ax.text((x1 + x2) / 2 + toff[0], (y1 + y2) / 2 + toff[1], text,
                ha="center", va="bottom", fontsize=fs, color="#111827")


def new_ax(w, h):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    import os
    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(f"{FIGDIR}/{name}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {FIGDIR}/{name}")


# ------------------------------------------------- 1. MMG 三自由度模型示意
def fig_mmg():
    fig, ax = new_ax(10, 6.4)
    ax.text(5, 9.55, "MMG 三自由度运动模型（dynamics.py）", ha="center", fontsize=13, weight="bold")

    # 坐标系与船体
    cx, cy = 2.6, 5.6
    ship = plt.Polygon([[cx + 1.5, cy], [cx - 1.0, cy + 0.55], [cx - 1.0, cy - 0.55]],
                       fc="#fca5a5", ec=EDGE, zorder=3)
    ax.add_patch(ship)
    arrow(ax, cx - 1.8, cy - 1.6, cx + 1.8, cy - 1.6, "", color="#6b7280")
    arrow(ax, cx - 1.8, cy - 1.6, cx - 1.8, cy + 1.2, "", color="#6b7280")
    ax.text(cx + 1.9, cy - 1.7, "x(北)", fontsize=10)
    ax.text(cx - 1.75, cy + 1.3, "y(东)", fontsize=10)
    arrow(ax, cx, cy, cx + 1.9, cy + 0.1, "u 纵荡", rad=-0.2, color="#2563eb")
    arrow(ax, cx, cy, cx - 0.1, cy + 1.3, "v 横荡", rad=0.2, color="#2563eb")
    ax.add_patch(FancyArrowPatch((cx + 0.9, cy + 0.75), (cx + 0.2, cy + 1.0), arrowstyle="-|>",
                                 mutation_scale=13, color="#2563eb", connectionstyle="arc3,rad=0.5"))
    ax.text(cx + 0.55, cy + 1.25, "r 艏摇", fontsize=10, color="#2563eb")
    ax.text(cx - 1.15, cy - 0.9, "δ 舵角 ±35°\nn 转速 0~25 rpm", fontsize=9, ha="left")

    # 方程
    box(ax, 5.1, 6.6, 4.6, 2.5,
        "水动力/力矩（_forces, dynamics.py:96）\n"
        "X = x_u·u + x_uu·|u|u + x_n·n|n|\n"
        "Y = y_v·v + y_r·r + y_vv·|v|v + y_δ·u²δ\n"
        "N = n_v·v + n_r·r + n_rr·|r|r + n_δ·u²δ", C_ALGO, fs=9)
    box(ax, 5.1, 3.5, 4.6, 2.5,
        "加速度（step, dynamics.py:119）\n"
        "u̇ = X/m_u + r·v + res_u\n"
        "v̇ = Y/m_v − r·u + res_v\n"
        "ṙ = N/i_z + res_r\n"
        "res: ResidualModel 残差补偿(5→64→64→3)", C_CORE, fs=9)
    box(ax, 5.1, 0.5, 4.6, 2.4,
        "积分为世界坐标（dt=0.1s 显式欧拉）\n"
        "ψ += r·dt\n"
        "ẋ = cosψ·(u+d_u) − sinψ·(v+d_v)\n"
        "ẏ = sinψ·(u+d_u) + cosψ·(v+d_v)\n"
        "d_u,d_v: 体坐标系流扰", C_ENV, fs=9)
    arrow(ax, 4.3, 5.6, 5.1, 5.6)
    ax.text(2.6, 2.6, "执行器：δ̇≤4°/s, ṅ≤2rpm/s（速率限幅）\n阶段2再叠加一阶滞后 lag=0.2s", fontsize=9,
            ha="center", style="italic", color="#4b5563")
    save(fig, "tfig1_mmg_model.png")


# ------------------------------------------------- 2. 课程学习三阶段
def fig_curriculum():
    fig, ax = new_ax(11, 5.6)
    ax.text(5, 9.5, "三阶段课程学习（env.py:93-133 + train_ship_3dof.py:143）",
            ha="center", fontsize=13, weight="bold")
    stages = [
        ("阶段 0（episode<300 / 第1期训练）\n"
         "MMG 参数随机 ±3%\n无流扰 · 路径曲率×0.65\n紧致初始(y_std=0.2m, ψ=1°)\n无传感器噪声 · 无执行器滞后", C_CORE),
        ("阶段 1（episode<900 / 第2期）\n"
         "MMG 参数随机 ±8%\n流扰 0.55× · 曲率×0.85\n初始放宽(y_std=0.35m, ψ=1.6°)\n无传感器噪声 · 无执行器滞后", C_ALGO),
        ("阶段 2（episode≥900 / 第3期）\n"
         "MMG 参数随机 ±15%\n全量流扰+随机游走漂移(±0.6)\n曲率×1.0 · 初始最宽\n传感器噪声开 · 执行器滞后0.2s", C_ENV),
    ]
    w, h, y = 2.9, 4.6, 3.0
    xs = [0.4, 3.6, 6.8]
    for x, (t, c) in zip(xs, stages):
        box(ax, x, y, w, h, t, c, fs=9)
    arrow(ax, xs[0] + w, y + h / 2, xs[1], y + h / 2, "60万步\n热启动", toff=(0, 0.3))
    arrow(ax, xs[1] + w, y + h / 2, xs[2], y + h / 2, "80万步\n热启动", toff=(0, 0.3))
    box(ax, 1.6, 0.6, 6.8, 1.6,
        "实现：train_ship_3dof.py 循环调 train.py 三次，\n"
        "每期用 --trained-agent 从上一期模型继续训练；\n"
        "环境侧由 --env-kwargs curriculum_stage:N 固定阶段", C_CLI, fs=9)
    arrow(ax, 5, 3.0, 5, 2.2)
    save(fig, "tfig2_curriculum.png")


# ------------------------------------------------- 3. SAC 网络结构
def fig_network():
    fig, ax = new_ax(11, 6)
    ax.text(5, 9.55, "SAC 智能体网络结构（policy_kwargs: net_arch=[512,512,512]）",
            ha="center", fontsize=13, weight="bold")
    box(ax, 0.3, 5.4, 2.0, 2.2, "观测 o (14维)\ne_y,e_ψ,e_s\nu,v,r,β,δ,n\nδ̇,ṅ,κ_l\nd̂_u,d̂_v", C_ENV, fs=8.5)
    box(ax, 2.9, 6.7, 3.4, 2.6, "Actor（策略网络 π_θ）\nMLP 14→512→512→512\n输出 μ, log_std (各2维)\n高斯采样+tanh 压缩\na=[δ̇_cmd, ṅ_cmd]∈[-1,1]²", C_ALGO, fs=9, weight="bold")
    box(ax, 2.9, 3.4, 3.4, 2.6, "Critic（双 Q 网络）\nMLP (14+2)→512→512→512\nQ₁(o,a), Q₂(o,a) 各1维\ntarget 取 min 抑制过估计\n熵正则 α 自动调节", C_CORE, fs=9, weight="bold")
    box(ax, 7.0, 5.4, 2.7, 2.2, "环境交互\nreplay buffer 1e6\ntrain_freq=4\ngradient_steps=4\nbatch=512, γ=0.99, τ=0.005", C_CLI, fs=8.5)
    arrow(ax, 2.3, 6.6, 2.9, 7.6)
    arrow(ax, 2.3, 6.1, 2.9, 5.0)
    arrow(ax, 6.3, 8.0, 7.0, 7.0, "动作 a")
    arrow(ax, 7.0, 6.0, 6.3, 4.6, "(o,a,r,o')\n样本", toff=(1.0, -0.6))
    box(ax, 3.6, 0.6, 5.2, 1.7,
        "关键超参（hyperparams/sac.yml:266 + 脚本覆盖）\n"
        "lr=3e-4 · ent_coef=auto · use_sde=True\n"
        "learning_starts=5000 · VecNormalize(norm_obs)", C_OUT, fs=9)
    arrow(ax, 5.5, 3.4, 5.5, 2.3)
    save(fig, "tfig3_network.png")


# ------------------------------------------------- 4. 奖励构成
def fig_reward():
    fig, ax = new_ax(11, 5.8)
    ax.text(5, 9.55, "奖励函数构成（env.py:195 _reward）", ha="center", fontsize=13, weight="bold")
    box(ax, 0.3, 6.8, 9.4, 1.6,
        "reward = r_progress − r_track − r_heading − r_smooth − r_energy − r_safety   (+50 到终点 / −100 失败)",
        C_OUT, fs=10.5, weight="bold")
    items = [
        ("r_progress 进度\n0.85·clip(Δs,0,1.25)\n沿路径前进给正奖励", C_CORE),
        ("r_track 横偏\n1.1·|e_y| + 0.35·e_y²\n偏离航线惩罚", C_ENV),
        ("r_heading 航向\n0.6·|e_ψ| + 0.08·r²\n航向偏差+艏摇惩罚", C_ENV),
        ("r_smooth 平滑\n0.03·δ̇² + 0.015·ṅ²\n抑制舵机抖动", C_ALGO),
        ("r_energy 能耗\n0.005·n²\n抑制高转速", C_ALGO),
        ("r_safety 安全\n3·碰撞 + 4·出航道\n(|e_y|>8m)", C_GATE),
    ]
    w, h, y = 1.5, 3.4, 2.6
    xs = [0.2 + i * 1.62 for i in range(6)]
    for x, (t, c) in zip(xs, items):
        box(ax, x, y, w, h, t, c, fs=8)
        arrow(ax, x + w / 2, y + h, x + w / 2, 6.8)
    ax.text(5, 1.6, "终止条件：到达 s≥final_s−1 (+50) | 碰撞 / |e_y|>12 / |r|>1.25 / u∉[−0.5,4] / 出航道 (−100) | step≥3000 截断",
            ha="center", fontsize=9, style="italic", color="#4b5563")
    save(fig, "tfig4_reward.png")


# ------------------------------------------------- 5. 真实训练曲线
def fig_training_curves():
    base = "/media/jim/jim_11/rl-zoo3-ship3dof-docker/logs_pipeline/ship3dof_1km/sac"
    phases = [("ShipPathTracking3DOF-v0_1", "阶段0 (60万步)"),
              ("ShipPathTracking3DOF-v0_2", "阶段1 (80万步)"),
              ("ShipPathTracking3DOF-v0_3", "阶段2 (105万步,中断)"),
              ("ShipPathTracking3DOF-v0_5", "阶段2续训 (10万步)")]
    colors = ["#2563eb", "#16a34a", "#ea580c", "#9333ea"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), dpi=150, sharex=True)

    t_offset = 0.0
    for (run, label), color in zip(phases, colors):
        files = sorted(glob.glob(f"{base}/{run}/0.monitor.csv"))
        if not files:
            continue
        rs, ls, ts = [], [], []
        with open(files[0]) as f:
            lines = [ln for ln in f if not ln.startswith("#")]
        for row in csv.DictReader(lines):
            rs.append(float(row["r"]))
            ls.append(float(row["l"]))
            ts.append(float(row["t"]))
        steps = np.cumsum(ls)
        x = steps + t_offset
        t_offset = x[-1]
        rs = np.clip(rs, -3000.0, None)  # 裁剪极端负回报离群值，避免压扁曲线

        def rolling(v, w=51):
            v = np.asarray(v)
            if len(v) < w:
                return v
            return np.convolve(v, np.ones(w) / w, mode="valid")

        xr = x[len(x) - len(rolling(rs)):]
        ax1.plot(x, rs, color=color, alpha=0.12, lw=0.5)
        ax1.plot(xr, rolling(rs), color=color, lw=1.8, label=label)
        ax2.plot(x, ls, color=color, alpha=0.15, lw=0.5)
        ax2.plot(xr, rolling(ls), color=color, lw=1.8, label=label)
        for a in (ax1, ax2):
            a.axvline(x[0], color="#9ca3af", ls=":", lw=0.8)

    ax1.set_ylabel("episode return（51回合滑动均值）")
    ax1.set_title("ShipPathTracking3DOF-v0 课程 SAC 训练曲线（1km 路径, 255万步, seed 42）")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(alpha=0.3)
    ax2.set_ylabel("episode length（步）")
    ax2.set_xlabel("累计训练步数")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "tfig5_training_curves.png")


# ------------------------------------------------- 6. 流水线（复用 fig3 风格，含本次修复说明）
def fig_fix():
    fig, ax = new_ax(10, 4.6)
    ax.text(5, 9.4, "关键修复：出生点对齐路径起点（env.py:123-143）", ha="center", fontsize=13, weight="bold")
    box(ax, 0.4, 4.6, 4.4, 3.4,
        "修复前\nstate.y ~ N(0, 0.2)\n路径横向偏移随机 ±20m\n→ 约1/3回合出生即 |e_y|>12\n一步终止(-100), 无法学习\nep_len_mean ≈ 9 步", C_ENV, fs=9)
    box(ax, 5.2, 4.6, 4.4, 3.4,
        "修复后\ny0 = path.y_at_s(0)\nstate.y ~ N(y0, y_std)\n出生即贴近路径起点\n初始 |e_y| < 0.8m\nep_len_mean ≈ 2000+ 步", C_CORE, fs=9)
    arrow(ax, 4.8, 6.3, 5.2, 6.3)
    box(ax, 1.2, 0.9, 7.6, 2.2,
        "同时配套：path.py 新增 y_at_s() 弧长插值；\n"
        "path_length 500→1000m, max_steps 1200→3000；\n"
        "评估必须加载 vecnormalize.pkl（norm_obs=True 训练）", C_ALGO, fs=9)
    save(fig, "tfig6_fix.png")


if __name__ == "__main__":
    fig_mmg()
    fig_curriculum()
    fig_network()
    fig_reward()
    fig_training_curves()
    fig_fix()
