# vessel_control: ONNX Runtime + OSQP

## Build (ROS)

```bash
cd bulkcarrier_ws
catkin build vessel_control
```

CMake options (default ON):

- `VESSEL_USE_OSQP` — fetch & link OSQP v0.6.3
- `VESSEL_USE_ONNXRUNTIME` — download ONNX Runtime 1.17.1 (linux x64)

## Standalone MPC test (no ROS)

```bash
# 1) Export a trained model
rosrun vessel_identification export_model.py --checkpoint runs/koopman/best.pt \
  --registry bulkcarrier_ws/src/vessel_control/model_registry \
  --model-id koopman_test --data data/processed/<dataset>

# 2) Build & run
cd bulkcarrier_ws/src/vessel_control/test
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/mpc_standalone_test ../model_registry/koopman_test
```

## Runtime

- Loads `meta.yaml`, `norm_*.json`, `encoder.onnx`, `A.bin`, `B.bin`, optional `tube_tightening.csv`
- Subscribes: `/ins/state`, `/trajectory/ref`, `/sensors/rudder_deg`, `/sensors/shaft_rpm`
- Publishes: `/control/cmd`, `/control/status`

MPC optimizes in **normalized** control space; outputs physical `delta_rad` and `rpm`.

## Hard constraints

| 类型 | 形式（物理量） | QP 中实现 |
|------|----------------|-----------|
| 舵角 / 转速 | \(u_{min} \le u_k \le u_{max}\) | 归一化盒约束 |
| 舵速 / 转速变化率 | \(\|u_k-u_{k-1}\| \le \dot u_{max} T_s\) | \(k=0\) 相对实测 `u_prev`；\(k\ge1\) 相邻步差分 |

参数来自 `meta.yaml` / `control_default.yaml`：`delta_rate_max` [rad/s]、`n_rate_max` [rpm/s]。

验证（无 ROS）：

```bash
./build/rate_constraint_test ../model_registry/koopman_test
```
