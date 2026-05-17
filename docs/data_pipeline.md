# Data pipeline

## Manifest (`data/manifests/*.yaml`)

| Field | Description |
|-------|-------------|
| `dataset_id` | Logical dataset name |
| `Ts` | Resample period [s], match MPC (default 0.25) |
| `sources` | List of `rosbag` paths |
| `topics` | INS / rudder / RPM / wind topic names |
| `output.root` | Processed output directory |
| `output.format` | `npz` or `csv` |
| `processing.rudder_in_degrees` | Convert rudder to rad in `u[:,0]` |
| `splits` | Chronological fractions (sum to 1.0) |

Paths are relative to the repository root (parent of `data/` when manifest lives in `data/manifests/`).

## Commands

```bash
# Manifest-driven
rosrun vessel_tools time_sync.py --manifest data/manifests/example_trial.yaml

# Single bag
rosrun vessel_tools bag2dataset.py --bag data/raw/trial.bag --out data/processed/trial1 --ts 0.25

# Without ROS (rosbags backend)
python3 bulkcarrier_ws/src/vessel_tools/scripts/time_sync.py \
  --manifest data/manifests/example_trial.yaml --backend rosbags
```

## Tests

```bash
cd bulkcarrier_ws/src/vessel_tools
PYTHONPATH=src python3 test/test_pipeline.py -v
```
