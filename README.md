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

rosrun vessel_tools bag2dataset.py --bag data/raw/trial.bag --out data/processed/run1
rosrun vessel_identification train.py --data data/processed/run1
rosrun vessel_identification export_model.py --checkpoint runs/koopman/best.pt \
  --registry bulkcarrier_ws/src/vessel_control/model_registry --model-id koopman_v20260517_001
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

## Status

Scaffold only — ONNX lift, OSQP MPC, MMG plant, and training loops are `TODO` stubs.
