#!/usr/bin/env python3
"""生成代码设计文档所需的架构图（PNG）。"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FIGDIR = "docs/design/figures"

C_CLI = "#dbeafe"      # 蓝
C_CORE = "#dcfce7"     # 绿
C_ALGO = "#fef3c7"     # 黄
C_ENV = "#fee2e2"      # 红
C_OUT = "#f3e8ff"      # 紫
C_GATE = "#ffedd5"     # 橙
EDGE = "#374151"


def box(ax, x, y, w, h, text, fc, fs=10, weight="normal", ec=EDGE):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.2, edgecolor=ec, facecolor=fc, zorder=2,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, weight=weight, zorder=3)


def arrow(ax, x1, y1, x2, y2, text="", fs=8.5, style="-|>", color=EDGE, rad=0.0, toff=(0, 0.12)):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle=style, mutation_scale=14,
            linewidth=1.3, color=color,
            connectionstyle=f"arc3,rad={rad}", zorder=1,
        )
    )
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


# ---------------------------------------------------------- fig1: 整体架构
def fig_architecture():
    fig, ax = new_ax(11.5, 7.5)
    ax.text(5, 9.65, "rl-baselines3-zoo（Ship3DOF 分叉版）整体分层架构",
            ha="center", fontsize=13, weight="bold")

    # CLI 层
    box(ax, 0.4, 8.0, 2.1, 1.1, "train.py / enjoy.py\n(根入口, 4 行转发)", C_CLI, fs=8.5)
    box(ax, 2.9, 8.0, 2.1, 1.1, "rl_zoo3/cli.py\nrl_zoo3 子命令分发", C_CLI, fs=8.5)
    box(ax, 5.4, 8.0, 2.1, 1.1, "benchmark.py\nrecord_video.py\npush_to_hub.py", C_CLI, fs=8)
    box(ax, 7.9, 8.0, 1.9, 1.1, "plots/\n绘图与基准报告", C_CLI, fs=8.5)

    # 核心层
    box(ax, 0.4, 5.6, 4.3, 1.7,
        "exp_manager.py  ExperimentManager (核心)\n读 yml 超参 → 预处理 → 建 VecEnv/回调\n→ 建/加载 SB3 模型 → learn → 保存\n+ Optuna 超参优化", C_CORE, fs=9, weight="bold")
    box(ax, 5.1, 5.6, 2.3, 1.7,
        "utils.py\nALGOS 注册表\nwrapper/callback 解析\n模型路径解析", C_CORE, fs=8.5)
    box(ax, 7.6, 5.6, 2.2, 1.7,
        "callbacks.py\nTrialEvalCallback\nParallelTrainCallback\nSaveVecNormalize…", C_CORE, fs=8)

    # 算法层
    box(ax, 0.4, 3.4, 4.6, 1.3,
        "Stable Baselines3:  A2C / DDPG / DQN / PPO / SAC / TD3\nsb3_contrib:  ARS / CrossQ / QRDQN / TQC / TRPO / PPO-LSTM", C_ALGO, fs=9)
    box(ax, 5.3, 3.4, 4.5, 1.3,
        "hyperparams/{algo}.yml\n按 env_id 分节存储调好的超参数\n(sac.yml:266 → ShipPathTracking3DOF-v0)", C_ALGO, fs=8.5)

    # 环境层
    box(ax, 0.4, 1.2, 4.6, 1.4,
        "Gymnasium 环境\n内置/MuJoCo/Atari/PyBullet…\nimport_envs.py 统一注册", C_ENV, fs=8.5)
    box(ax, 5.3, 1.2, 4.5, 1.4,
        "custom_envs/ship_3dof (本分叉新增)\nShipPathTracking3DOF-v0\nMMG 动力学 + 课程学习 + 部署安全件", C_ENV, fs=8.5, weight="bold")

    arrow(ax, 1.5, 8.0, 2.0, 7.3)
    arrow(ax, 4.0, 8.0, 3.2, 7.3)
    arrow(ax, 6.4, 8.0, 6.2, 7.3)
    arrow(ax, 8.8, 8.0, 8.7, 7.3)
    arrow(ax, 2.5, 5.6, 2.5, 4.7)
    arrow(ax, 7.5, 5.6, 7.6, 4.7)
    arrow(ax, 2.5, 3.4, 2.5, 2.6)
    arrow(ax, 7.5, 3.4, 7.5, 2.6)
    arrow(ax, 5.0, 1.9, 5.3, 1.9, "gym.register")

    ax.text(0.25, 8.55, "入口层", rotation=90, fontsize=9, color="#6b7280")
    ax.text(0.25, 6.35, "核心层", rotation=90, fontsize=9, color="#6b7280")
    ax.text(0.25, 3.95, "算法/配置层", rotation=90, fontsize=9, color="#6b7280")
    ax.text(0.25, 1.8, "环境层", rotation=90, fontsize=9, color="#6b7280")
    save(fig, "fig1_architecture.png")


# ---------------------------------------------------------- fig2: 训练流程
def fig_training_flow():
    fig, ax = new_ax(9, 10.5)

    ax.text(3.6, 9.7, "训练调用流程（train.py → 模型落盘）", ha="center", fontsize=13, weight="bold")

    steps = [
        ("python train.py --algo sac --env ShipPathTracking3DOF-v0", C_CLI),
        ("rl_zoo3.train.train(): 解析参数\nimport_envs 注册环境 / 校验 env_id / 设种子", C_CLI),
        ("ExperimentManager.setup_experiment()", C_CORE),
        ("read_hyperparameters: hyperparams/sac.yml\n(env_id 精确匹配 → atari → default 回退; CLI 覆盖)", C_ALGO),
        ("_preprocess_hyperparams: schedule/normalize\n/policy_kwargs/wrapper/callback 解析", C_ALGO),
        ("create_envs: make_vec_env + VecNormalize\ncreate_callbacks: Eval/Checkpoint/ProgressBar", C_ENV),
        ("ALGOS['sac'](**hyperparams) 建模\n或 --trained-agent 断点续训(课程学习用)", C_ALGO),
        ("model.learn(n_timesteps)\nEvalCallback 周期评估, 保存 best_model", C_CORE),
        ("save_trained_model: {env}.zip + vecnormalize.pkl\n+ config.yml / args.yml / command.txt", C_OUT),
    ]
    w, h = 5.4, 0.82
    x = 1.0
    ys = [8.55 - i * 1.02 for i in range(len(steps))]
    for y, (t, c) in zip(ys, steps):
        box(ax, x, y, w, h, t, c, fs=8.5)
    for i in range(len(ys) - 1):
        arrow(ax, x + w / 2, ys[i], x + w / 2, ys[i + 1] + h)

    # 旁路: Optuna
    box(ax, 7.3, 4.4, 2.4, 1.9,
        "--optimize-hyperparameters\nOptuna study\nsample_*_params 采样\nTrialEvalCallback 剪枝", C_GATE, fs=8)
    arrow(ax, x + w, ys[2] + h / 2, 7.3, 5.35, rad=-0.2)
    arrow(ax, 8.5, 4.4, x + w, ys[6] + h / 2, rad=-0.2)

    ax.text(3.6, 0.12, "输出目录: logs/{algo}/{env}_{run_id}/", ha="center",
            fontsize=9, style="italic", color="#4b5563")
    save(fig, "fig2_training_flow.png")


# ---------------------------------------------------------- fig5: 部署形态
def fig_deployment():
    fig, ax = new_ax(11.5, 6.5)
    ax.text(5, 9.6, "Docker / K8s 部署形态", ha="center", fontsize=13, weight="bold")

    # 镜像层
    box(ax, 0.3, 6.6, 3.0, 2.2,
        "ship3dof:latest\nDockerfile.ship3dof\npytorch 2.5.1+cuDNN9\nCUDA 12.4 训练镜像\n(纯训练, CMD bash)", C_ALGO, fs=8.5)
    box(ax, 3.6, 6.6, 3.0, 2.2,
        "ship3dof-remote:latest\nDockerfile.remote\nVNC/noVNC + Jupyter\n+ TensorBoard + SSH\n(supervisord 常驻)", C_ALGO, fs=8.5)
    box(ax, 6.9, 6.6, 2.9, 2.2,
        "isaac-lab-ssh:v3.0\nDockerfile.isaac-ssh\nIsaac Sim 自带 python\nJupyter + TB + SSH", C_ALGO, fs=8.5)

    # 运行形态
    box(ax, 0.3, 3.4, 4.5, 2.0,
        "本机 Docker\ndocker run --gpus all\n-v 项目:/workspace/rl-zoo\n挂载盘持久化 logs/artifacts", C_CLI, fs=8.5)
    box(ax, 5.1, 3.4, 4.7, 2.0,
        "docker-compose.yml\nship3dof-remote (默认): 6080/8888/6006/2222\nship3dof-train (profile=train): 一次性训练", C_CLI, fs=8.5)

    # K8s
    box(ax, 0.3, 0.5, 9.5, 2.0,
        "K8s (namespace ship3dof):  deployment-remote 远程桌面 | service + ingress (noVNC/Jupyter/TB 子路径)\n"
        "job-train (单管线) | job-multi-gpu (多卡并行) | job-autotune (自动调参) | PVC: logs/artifacts/autotune",
        C_CORE, fs=8.5)

    arrow(ax, 1.8, 6.6, 2.2, 5.4)
    arrow(ax, 5.1, 6.6, 7.0, 5.4)
    arrow(ax, 2.5, 3.4, 3.5, 2.5)
    arrow(ax, 7.4, 3.4, 6.5, 2.5)
    save(fig, "fig5_deployment.png")


# ---------------------------------------------------------------- fig3: 流水线
def fig_pipeline():
    fig, ax = new_ax(12, 6.2)

    ax.text(5, 9.55, "Ship-3DOF Sim2Real 训练流水线（run_known_mmg_pipeline.py）",
            ha="center", fontsize=13, weight="bold")

    steps = [
        ("试验数据\ngenerate_mmg_\ntrials.py\n(合成/实船 CSV)", C_OUT),
        ("M1 门控\ncheck_mmg_\nbaseline.py\nMMG 模型回放验证\nRMSE ψ≤5°, r≤0.1", C_GATE),
        ("M2 门控\ntrain_residual.py\n残差网络训练\n验证损失下降 ≥0.2", C_GATE),
        ("课程式 SAC 训练\ntrain_ship_3dof.py\n3 阶段课程学习\n域随机→扰动→噪声", C_ALGO),
        ("M3 门控\nevaluate_ship_\n3dof.py\n随机仿真评估\n成功率 ≥0.95, 碰撞=0", C_GATE),
        ("M4 门控\nrun_shadow_\nmode.py\n影子模式安全监督\n接管率 ≤0.02", C_GATE),
        ("决策\ngo_for_limited\n_takeover\npipeline_report.json", C_CORE),
    ]
    w, h, y = 1.22, 2.6, 4.6
    xs = [0.15 + i * 1.42 for i in range(len(steps))]
    for x, (t, c) in zip(xs, steps):
        box(ax, x, y, w, h, t, c, fs=7.5)
    for i in range(len(xs) - 1):
        arrow(ax, xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)

    # 下方支撑模块
    box(ax, 0.9, 1.2, 3.6, 1.9,
        "multi_gpu_train.py\n多 GPU 调度器：一任务一 GPU\n(CUDA_VISIBLE_DEVICES 绑定,\nThreadPoolExecutor 并行, 多种子)", C_CLI, fs=8.5)
    box(ax, 5.4, 1.2, 3.6, 1.9,
        "autotune_ship_3dof.py\n时间预算超参搜索\n并行发起 pipeline 作业\n按 balanced_score 排名", C_CLI, fs=8.5)
    arrow(ax, 2.7, 3.1, 3.4, y - 0.05, rad=0.15)
    arrow(ax, 7.2, 3.1, 6.6, y - 0.05, rad=-0.15)
    ax.text(5, 0.35, "产物：artifacts/ship_3dof/runs/<run-tag>/  (各门控 JSON 报告 + 模型 + pipeline_report.json)",
            ha="center", fontsize=9, style="italic", color="#4b5563")
    save(fig, "fig3_pipeline.png")


# ------------------------------------------------------- fig4: 环境模块结构
def fig_env_modules():
    fig, ax = new_ax(11, 7)

    ax.text(5, 9.6, "custom_envs/ship_3dof 模块结构", ha="center", fontsize=13, weight="bold")

    box(ax, 3.5, 6.9, 3.0, 1.9,
        "env.py\nShipPathTracking3DOFEnv\nobs(14) / act(2)\n课程学习 / 奖励 / 终止判定", C_ENV, fs=9, weight="bold")

    box(ax, 0.3, 4.0, 2.6, 1.8,
        "dynamics.py\nMMGDynamics (MMG 模型)\nShipState(8) / Euler dt=0.1s\nResidualModel (tanh MLP)", C_ALGO, fs=8.5)
    box(ax, 3.7, 4.0, 2.6, 1.8,
        "path.py\nReferencePath\n500m 双谐波路径\n最近点/横向误差/曲率", C_ALGO, fs=8.5)
    box(ax, 7.1, 4.0, 2.6, 1.8,
        "config.py\nMMGParameters(14 系数)\nActuatorLimits / RewardWeights\nEnvironmentConfig", C_ALGO, fs=8.5)

    box(ax, 0.3, 0.7, 4.2, 2.0,
        "identification.py  系统辨识\nread_trial_csv / preprocess_trial\nfit_nomoto / fit_mmg_least_squares\nrefine_mmg_multiple_shooting", C_CORE, fs=8.5)
    box(ax, 5.5, 0.7, 4.2, 2.0,
        "deploy.py  部署安全\nSafetyFilter (动作裁剪/超限清零)\nFallbackPIDController\nShadowModeSupervisor", C_CORE, fs=8.5)

    arrow(ax, 1.6, 5.8, 4.2, 6.9, "u_dot,v_dot,r_dot", rad=0.1)
    arrow(ax, 5.0, 5.8, 5.0, 6.9, "e_y, e_ψ, e_s, κ")
    arrow(ax, 8.4, 5.8, 5.8, 6.9, "配置/权重", rad=-0.1)
    arrow(ax, 2.4, 2.7, 1.6, 4.0, "辨识参数 JSON", rad=0.1)
    arrow(ax, 7.6, 2.7, 6.0, 6.85, "影子模式包装策略", rad=-0.12)

    ax.text(5, 0.15, "注册: custom_envs/__init__.py → gymnasium id 'ShipPathTracking3DOF-v0', max_episode_steps=3000",
            ha="center", fontsize=9, style="italic", color="#4b5563")
    save(fig, "fig4_env_modules.png")


if __name__ == "__main__":
    fig_architecture()
    fig_training_flow()
    fig_pipeline()
    fig_env_modules()
    fig_deployment()
