# Deep Koopman training & export

## Model

- Encoder \(\phi\): MLP, `x_norm` → `z` (\(n_z=64\))
- Global linear dynamics: \(z_{k+1} = A z_k + B u_k\)
- Decoder (training only / `Cx.bin`): \(x \approx C z\)

## Loss

\[
\mathcal{L} = w_{ms}\sum_{i=0}^{N}\|\hat z_i - \phi(x_i)\|^2
+ w_{1}\|z_1 - (Az_0 + Bu_0)\|^2
+ w_{rec}\sum_i \|\psi(z_i)-x_i\|^2
\]

## Commands

```bash
rosrun vessel_identification train.py \
  --config $(rospack find vessel_identification)/config/koopman_default.yaml \
  --data data/processed/ship_trials_example \
  --out runs/koopman

rosrun vessel_identification export_model.py \
  --checkpoint runs/koopman/best.pt \
  --registry $(rospack find vessel_control)/model_registry \
  --model-id koopman_v20260517_001 \
  --data data/processed/ship_trials_example

# PyTorch checkpoint -> ONNX only (verified Torch vs ORT parity)
rosrun vessel_identification convert_to_onnx.py \
  --checkpoint runs/koopman/best.pt \
  --out-dir /tmp/onnx_export

# Full deploy bundle (ONNX + A/B + encoder_io.json + meta.yaml)
rosrun vessel_identification convert_to_onnx.py \
  --checkpoint runs/koopman/best.pt \
  --full-bundle \
  --registry $(rospack find vessel_control)/model_registry \
  --model-id koopman_v20260517_001 \
  --data data/processed/ship_trials_example
```

## Export bundle

| File | Content |
|------|---------|
| `encoder.onnx` | \(\phi(x_{norm})\) |
| `encoder_io.json` | ONNX Runtime tensor names and dimensions for C++/SIL |
| `A.bin`, `B.bin` | `float64` row-major |
| `Cx.bin` | Decoder weights `[nx, nz]` |
| `norm_x.json`, `norm_u.json` | Standardization |
| `tube_tightening.csv` | From horizon validation |
| `meta.yaml`, `checksums.sha256` | Deploy contract |

## Tests

```bash
cd bulkcarrier_ws/src/vessel_identification
PYTHONPATH=src python3 test/test_koopman.py -v
```
