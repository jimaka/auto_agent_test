"""Python Koopman MPC for SIL (ONNX + OSQP), aligned with vessel_control."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import osqp
    import scipy.sparse as sp
except ImportError:
    osqp = None
    sp = None

from vessel_simulation.mmg3dof import ControlInput, VesselState, wrap_pi


class KoopmanMpcController:
    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir)
        self._load_bundle()
        self._build_prediction()
        self._solver: Optional[osqp.OSQP] = None

    def _load_bundle(self) -> None:
        if ort is None or osqp is None:
            raise ImportError("SIL Koopman MPC requires onnxruntime and osqp")

        meta_path = self.model_dir / "meta.yaml"
        text = meta_path.read_text(encoding="utf-8")
        self.nz = int(self._yaml_scalar(text, "nz", 64))
        self.N = int(self._yaml_scalar(text, "horizon_N", 30))
        self.Ts = float(self._yaml_scalar(text, "Ts", 0.25))
        self.nx, self.nu = 6, 2

        with (self.model_dir / "norm_x.json").open("r", encoding="utf-8") as f:
            nx = json.load(f)
        with (self.model_dir / "norm_u.json").open("r", encoding="utf-8") as f:
            nu = json.load(f)
        self.mu_x = np.asarray(nx["mu"], dtype=np.float64)
        self.sig_x = np.maximum(np.asarray(nx["sigma"], dtype=np.float64), 1e-6)
        self.mu_u = np.asarray(nu["mu"], dtype=np.float64)
        self.sig_u = np.maximum(np.asarray(nu["sigma"], dtype=np.float64), 1e-6)

        self.A = np.fromfile(self.model_dir / "A.bin", dtype=np.float64).reshape(self.nz, self.nz)
        self.B = np.fromfile(self.model_dir / "B.bin", dtype=np.float64).reshape(self.nz, self.nu)
        cx_path = self.model_dir / "Cx.bin"
        if cx_path.is_file():
            self.Cx = np.fromfile(cx_path, dtype=np.float64).reshape(self.nx, self.nz)
        else:
            self.Cx = np.eye(self.nx, self.nz)

        io_path = self.model_dir / "encoder_io.json"
        if io_path.is_file():
            with io_path.open("r", encoding="utf-8") as f:
                io_spec = json.load(f)
            self.encoder_input = io_spec.get("input_name", "x_norm")
            self.encoder_output = io_spec.get("output_name", "z")
            self.nx = int(io_spec.get("nx", self.nx))
            self.nz = int(io_spec.get("nz", self.nz))
        else:
            self.encoder_input, self.encoder_output = "x_norm", "z"

        self.session = ort.InferenceSession(
            str(self.model_dir / "encoder.onnx"), providers=["CPUExecutionProvider"]
        )

        self.delta_min = np.deg2rad(-35.0)
        self.delta_max = np.deg2rad(35.0)
        self.n_min, self.n_max = 0.0, 120.0
        self.delta_rate = np.deg2rad(2.0)
        self.n_rate = 5.0

    @staticmethod
    def _yaml_scalar(text: str, key: str, default: float) -> float:
        import re

        m = re.search(rf"^\s*{key}\s*:\s*([0-9eE+\.-]+)\s*$", text, re.MULTILINE)
        return float(m.group(1)) if m else default

    def _norm_x(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mu_x) / self.sig_x

    def _norm_u(self, u: np.ndarray) -> np.ndarray:
        return (u - self.mu_u) / self.sig_u

    def _denorm_u(self, u_n: np.ndarray) -> np.ndarray:
        return u_n * self.sig_u + self.mu_u

    def lift(self, state: VesselState) -> np.ndarray:
        x = self._norm_x(state.as_vector()).astype(np.float32)
        z = self.session.run(
            None, {self.encoder_input: x.reshape(1, -1)}
        )[0].reshape(-1)
        return z[: self.nz]

    def _build_prediction(self) -> None:
        nz, nu, N = self.nz, self.nu, self.N
        Apow = [np.eye(nz)]
        for _ in range(N):
            Apow.append(self.A @ Apow[-1])
        self.Phi = np.vstack([Apow[k + 1] for k in range(N)])

        blocks = []
        for i in range(N):
            row = []
            for j in range(N):
                if j <= i:
                    row.append(Apow[i - j] @ self.B)
                else:
                    row.append(np.zeros((nz, nu)))
            blocks.append(np.hstack(row))
        self.Psi = np.vstack(blocks)

        Qx = np.diag([10.0, 10.0, 5.0, 1.0, 0.1, 0.1])
        self.Qz = self.Cx.T @ Qx @ self.Cx
        self.R = np.diag([0.1, 0.01])

    def compute(self, state: VesselState, u_prev: ControlInput, x_refs: List[VesselState]) -> ControlInput:
        z0 = self.lift(state)
        N, nu = self.N, self.nu
        n = N * nu

        z_refs = []
        for k in range(N):
            ref = x_refs[min(k, len(x_refs) - 1)]
            z_refs.append(self.lift(ref))
        z_ref = np.concatenate(z_refs)

        Qbar = np.kron(np.eye(N), self.Qz)
        Rbar = np.kron(np.eye(N), self.R)
        H = self.Psi.T @ Qbar @ self.Psi + Rbar
        f = self.Psi.T @ Qbar @ (self.Phi @ z0 - z_ref)

        du_delta = (self.delta_rate * self.Ts) / self.sig_u[0]
        du_n = (self.n_rate * self.Ts) / self.sig_u[1]

        u_prev_n = self._norm_u(np.array([u_prev.delta_rad, u_prev.rpm]))
        l_box = np.tile(self._norm_u(np.array([self.delta_min, self.n_min])), N)
        u_box = np.tile(self._norm_u(np.array([self.delta_max, self.n_max])), N)

        rows, cols, data = [], [], []
        l_vec, u_vec = [], []
        row = 0

        for k in range(N):
            for j in range(nu):
                c = k * nu + j
                rows.append(row)
                cols.append(c)
                data.append(1.0)
                l_vec.append(l_box[c])
                u_vec.append(u_box[c])
                row += 1

        for j in range(nu):
            du = du_delta if j == 0 else du_n
            rows.append(row)
            cols.append(j)
            data.append(1.0)
            l_vec.append(u_prev_n[j] - du)
            u_vec.append(u_prev_n[j] + du)
            row += 1

        for k in range(1, N):
            for j in range(nu):
                du = du_delta if j == 0 else du_n
                c, cm = k * nu + j, (k - 1) * nu + j
                rows += [row, row]
                cols += [c, cm]
                data += [1.0, -1.0]
                l_vec.append(-1e20)
                u_vec.append(du)
                row += 1
                rows += [row, row]
                cols += [c, cm]
                data += [1.0, -1.0]
                l_vec.append(-du)
                u_vec.append(1e20)
                row += 1

        A = sp.csc_matrix((data, (rows, cols)), shape=(row, n))
        P = sp.csc_matrix((H + H.T) / 2.0 + 1e-6 * sp.eye(n))

        prob = osqp.OSQP()
        prob.setup(P=P, q=f, A=A, l=np.array(l_vec), u=np.array(u_vec), verbose=False, polish=True)
        res = prob.solve()
        if res.info.status_val not in (1, 2):
            return u_prev
        u0_n = res.x[:nu]
        u0 = self._denorm_u(u0_n)
        return ControlInput(delta_rad=float(u0[0]), rpm=float(u0[1]))
