# Python ↔ C++ model bundle

Export directory: `vessel_control/model_registry/<model_id>/`

| File | Description |
|------|-------------|
| `meta.yaml` | Schema version, dimensions, weights, constraints |
| `A.bin` | `float64[64,64]` row-major |
| `B.bin` | `float64[64,2]` row-major |
| `encoder.onnx` | Normalized `x` → `z` (float32, opset 18) |
| `encoder_io.json` | ORT I/O contract: `input_name`, `output_name`, `nx`, `nz`, `opset` |
| `norm_x.json`, `norm_u.json` | Standardization |
| `tube_tightening.csv` | Per-step tightening |
| `checksums.sha256` | Integrity gate before deploy |

See `model_registry/_template/meta.yaml` for the full contract.
