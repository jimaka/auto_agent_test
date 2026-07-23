# AGENTS.md

## Cursor Cloud specific instructions

### Overview

This is a **Bulk Carrier Koopman-MPC** project: a ROS1 workspace for deep Koopman system identification (Python/PyTorch) and tube MPC tracking control (C++/OSQP) of a 65m bulk carrier vessel. The offline Python pipeline (training, export, simulation) runs independently of ROS.

### Running tests

All Python tests run without ROS. Each package uses `PYTHONPATH=src` with cross-package imports:

```bash
source /workspace/.venv/bin/activate

# vessel_tools (bag→dataset pipeline)
cd bulkcarrier_ws/src/vessel_tools && PYTHONPATH=src python3 -m unittest discover -s test -v

# vessel_identification (Koopman train/validate/export)
cd bulkcarrier_ws/src/vessel_identification && PYTHONPATH=src python3 -m unittest discover -s test -v

# vessel_simulation (MMG SIL benchmark — needs cross-package imports)
cd bulkcarrier_ws/src/vessel_simulation && PYTHONPATH=src:../vessel_identification/src:../vessel_tools/src python3 -m unittest discover -s test -v
```

### C++ standalone tests

The C++ standalone test (`bulkcarrier_ws/src/vessel_control/test/`) builds with CMake and auto-fetches OSQP v0.6.3 and ONNX Runtime 1.17.1. Requires `libeigen3-dev` and `libstdc++-14-dev`.

**Known issue:** `mpc_osqp.cpp:186` has a type mismatch (`double*` passed where `const std::vector<double>&` expected) that prevents compilation. This is a pre-existing code bug unrelated to environment setup.

### Key notes

- The venv lives at `/workspace/.venv`. Always activate it before running Python code.
- No ROS1 Noetic is available in this environment (Ubuntu 24.04 vs ROS Noetic's Ubuntu 20.04 target). The offline Python pipeline and simulation are fully functional without ROS.
- The `vessel_simulation` test requires PYTHONPATH to include both `vessel_identification/src` and `vessel_tools/src` for cross-package imports.
- PyTorch ONNX export uses the legacy TorchScript path (deprecation warnings are expected but harmless).
