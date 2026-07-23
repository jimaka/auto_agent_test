# MMG SIL 闭环基准测试

## 概述

`vessel_simulation` 提供 **MMG 三自由度** 船模 + 时间参数化参考轨迹 + **Koopman-MPC / Nomoto** 控制器对比，输出与设计一致的跟踪指标。

## 指标

| 指标 | 说明 |
|------|------|
| `e_cross_rms` | 横向误差 RMS [m] |
| `e_psi_rms` | 航向误差 RMS [rad] |
| `e_u_rms` | 纵向速度误差 RMS [m/s] |
| `max_overshoot_cross` | 横向误差最大绝对值 |
| `max_overshoot_u` | 速度误差最大绝对值 |

通过阈值（默认）：`e_cross_rms < 15 m`，`e_psi_rms < 0.5 rad`，`e_u_rms < 2 m/s`。

## 运行

```bash
pip install numpy pyyaml onnxruntime osqp scipy

rosrun vessel_simulation run_closed_loop.py \
  --scenario straight \
  --duration 120 \
  --model $(rospack find vessel_control)/model_registry/koopman_test \
  --out runs/sil_benchmark

# 可选：回写 meta.yaml 的 sil_passed
rosrun vessel_simulation run_closed_loop.py --scenario turn --duration 180 --update-meta
```

## 场景

- `straight` — 直线匀速
- `turn` — 匀速转弯
- `zigzag` — 分段变向

## 输出

`--out/<scenario>/benchmark_report.json` 含 Koopman 与 Nomoto 指标对比及 `koopman_passed`。

## 测试

```bash
cd bulkcarrier_ws/src/vessel_simulation
PYTHONPATH=src python3 test/test_sil.py -v
```
