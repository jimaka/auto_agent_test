# Bulk Carrier Koopman-MPC

ROS1 (Ubuntu) workspace for **deep Koopman identification** (Python) and **tube MPC tracking** (C++/OSQP) of a 65 m × 12.6 m bulk carrier.

## Layout

```text
bulkcarrier_ws/src/
  vessel_msgs          # ROS messages
  vessel_tools         # bag → dataset, sync, benchmarks
  vessel_identification# train / validate / export
  vessel_simulation    # MMG 3-DOF SIL
  vessel_control       # Koopman + MPC node
  vessel_baseline      # Nomoto fallback
  vessel_bringup       # stack / record launches
config/                # ship & MPC defaults
data/manifests/        # dataset versioning
docs/                  # architecture & interfaces
```

## Quick start (ROS1)

```bash
# Dependencies (example, Ubuntu 20.04 + Noetic)
sudo apt install ros-noetic-desktop-full python3-catkin-tools libeigen3-dev

cd bulkcarrier_ws
catkin init  # once
catkin build
source devel/setup.bash

roslaunch vessel_bringup stack.launch
# Fallback only:
roslaunch vessel_bringup stack.launch use_baseline:=true
```

## Python (offline)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Single bag -> NPZ (Ts=0.25 s, rudder deg->rad)
rosrun vessel_tools bag2dataset.py --bag data/raw/trial.bag --out data/processed/run1

# Manifest pipeline (multi-bag, train/val/test splits)
rosrun vessel_tools time_sync.py --manifest data/manifests/example_trial.yaml

rosrun vessel_identification train.py --data data/processed/ship_trials_example --out runs/koopman
rosrun vessel_identification validate_horizon.py --checkpoint runs/koopman/best.pt \
  --data data/processed/ship_trials_example
rosrun vessel_identification export_model.py --checkpoint runs/koopman/best.pt \
  --registry bulkcarrier_ws/src/vessel_control/model_registry \
  --model-id koopman_v20260517_001 --data data/processed/ship_trials_example
```

## Topics

| Topic | Message |
|-------|---------|
| `/ins/state` | `vessel_msgs/VesselState` |
| `/sensors/rudder_deg` | `std_msgs/Float64` (driver) |
| `/sensors/shaft_rpm` | `std_msgs/Float64` |
| `/sensors/wind` | `vessel_msgs/Wind` |
| `/trajectory/ref` | `vessel_msgs/TrajectoryRef` |
| `/control/cmd` | `vessel_msgs/ControlCmd` |
| `/control/status` | `vessel_msgs/ControlStatus` |

## Model registry

Template: `bulkcarrier_ws/src/vessel_control/model_registry/_template/`

Trained bundles are gitignored; copy template fields from `meta.yaml` when exporting.

## Data pipeline output

Processed datasets contain `train.npz`, `val.npz`, `test.npz` (when splits are configured), plus `meta.json` and a copy of `manifest.yaml`. Arrays:

| Key | Shape | Description |
|-----|-------|-------------|
| `t` | `[T]` | Time [s] |
| `x` | `[T, 6]` | `x,y,psi,u,v,r` |
| `u` | `[T, 2]` | `delta_rad, rpm` |
| `d` | `[T, 2]` | Wind speed/dir (optional) |

Bag reading uses `rosbag` when ROS is sourced, otherwise `rosbags` (`pip install rosbags`).

## Status

- **Done:** `vessel_tools` bag→dataset + manifest pipeline
- **Done:** `vessel_identification` Deep Koopman train / validate / ONNX export
- **Done:** `vessel_control` ONNX Runtime encoder + OSQP MPC (see `docs/control_mpc.md`)
- **Done:** `vessel_simulation` MMG SIL 闭环基准（见 `docs/sil_benchmark.md`）
- **TODO:** 实船 ROS 联调与海试
