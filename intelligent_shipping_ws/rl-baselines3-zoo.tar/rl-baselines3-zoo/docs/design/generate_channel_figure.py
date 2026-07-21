#!/usr/bin/env python3
"""生成"航道环境搭建流程"配图：左-搭建流程，右-示例航道地图。"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FIGDIR = "docs/design/figures"
C_CLI, C_CORE, C_ALGO, C_ENV, C_GATE = "#dbeafe", "#dcfce7", "#fef3c7", "#fee2e2", "#ffedd5"
EDGE = "#374151"


def box(ax, x, y, w, h, text, fc, fs=9, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=1.2, edgecolor=EDGE, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight, zorder=3)


def arrow(ax, x1, y1, x2, y2, color=EDGE):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.2, color=color, zorder=1))


fig = plt.figure(figsize=(13, 6.2), dpi=150)
gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.5], wspace=0.08)

# ---------------- 左：搭建流程 ----------------
ax = fig.add_subplot(gs[0])
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
ax.text(5, 9.7, "航道环境搭建流程", ha="center", fontsize=13, weight="bold")

steps = [
    ("① 船舶水动力参数\nmmg_params_example.json\n(或 identify_mmg.py 实船辨识)", C_ALGO),
    ("② 残差模型（可选）\ntrain_residual.py → residual_model.npz\nM2 门控 ≥0.2", C_CORE),
    ("③ 航道几何与边界\npath_length/curvature/waypoint_step\nchannel_half_width=8, fail=12\nobstacles=[(x,y,r),...]", C_ENV),
    ("④ 扰动与传感器\ndisturbance_scale / sensor_noise_std\nactuator_lag=0.2s", C_GATE),
    ("⑤ 实例化与冒烟验证\ngym.make('ShipPathTracking3DOF-v0', **kwargs)\npytest tests/test_ship_3dof_env.py", C_CLI),
]
y = 8.2
for text, c in steps:
    box(ax, 0.7, y - 1.15, 8.6, 1.45, text, c, fs=8.5)
    if y < 8.0:
        pass
    y -= 1.72
for i in range(4):
    arrow(ax, 5, 8.2 - i * 1.72 - 0.0 + 0.0, 5, 8.2 - i * 1.72 - 0.27)
# 修正箭头：从每个框底到下一个框顶
for i in range(4):
    yb = 8.2 - i * 1.72 - 1.15
    yt = 8.2 - (i + 1) * 1.72 + 0.30
    ax.add_patch(FancyArrowPatch((5, yb), (5, yt), arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.2, color=EDGE, zorder=1))

# ---------------- 右：示例航道 ----------------
ax2 = fig.add_subplot(gs[1])
data = json.load(open("/tmp/sample_path.json"))
px, py = np.array(data["x"]), np.array(data["y"])
ax2.plot(px, py, color="#6b7280", lw=1.8, label="参考路径 (ReferencePath)")
ax2.plot(px, py + 8, color="#d1d5db", lw=1.0, ls="--")
ax2.plot(px, py - 8, color="#d1d5db", lw=1.0, ls="--", label="航道边界 ±8m")
ax2.plot(px, py + 12, color="#fca5a5", lw=0.8, ls=":")
ax2.plot(px, py - 12, color="#fca5a5", lw=0.8, ls=":", label="失败线 ±12m")

# 障碍物
for ox, oy, r in [(420.0, float(np.interp(420, px, py)) + 4.0, 5.0),
                  (700.0, float(np.interp(700, px, py)) - 5.0, 4.0)]:
    ax2.add_patch(Circle((ox, oy), r, fc="#fecaca", ec="#dc2626", lw=1.2, zorder=3))
ax2.text(420, float(np.interp(420, px, py)) + 12.5, "障碍物 obstacles=[(x,y,r)]",
         fontsize=8.5, ha="center", color="#b91c1c")

# 出生点与船
y0 = py[0]
ship = Polygon([[18, y0], [2, y0 + 4], [2, y0 - 4]], fc="#dc2626", ec="#7f1d1d", zorder=5)
ax2.add_patch(ship)
ax2.text(10, y0 - 9, "出生点 y0=path.y_at_s(0)\n±y_std 噪声", fontsize=8.5, ha="left")

ax2.annotate("", xy=(950, float(np.interp(950, px, py))), xytext=(880, float(np.interp(880, px, py))),
             arrowprops=dict(arrowstyle="-|>", color="#16a34a", lw=2))
ax2.text(915, float(np.interp(915, px, py)) + 8, "终点 s≥final_s−1 → +50", fontsize=8.5,
         ha="center", color="#16a34a")

ax2.set_title("示例航道（1000m 双谐波路径 + 边界 + 障碍物）", fontsize=11)
ax2.set_xlabel("x [m]")
ax2.set_ylabel("y [m]")
ax2.legend(fontsize=8, loc="lower right")
ax2.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(f"{FIGDIR}/tfig8_channel_env.png", bbox_inches="tight", facecolor="white")
print(f"saved {FIGDIR}/tfig8_channel_env.png")
