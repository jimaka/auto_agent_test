from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from custom_envs.ship_3dof.config import MMGParameters
from custom_envs.ship_3dof.dynamics import MMGDynamics, ShipState

try:
    from scipy import optimize, signal
except ImportError:  # pragma: no cover - optional dependency
    optimize = None
    signal = None


REQUIRED_COLUMNS = ("t", "x", "y", "psi", "u", "v", "r", "delta", "n")


@dataclass
class TrialData:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    psi: np.ndarray
    u: np.ndarray
    v: np.ndarray
    r: np.ndarray
    delta: np.ndarray
    n: np.ndarray


@dataclass
class IdentificationResult:
    nomoto_k: float
    nomoto_t: float
    mmg_params: MMGParameters
    validation: dict[str, float]


def read_trial_csv(path: str | Path) -> TrialData:
    path = Path(path)
    rows: list[dict[str, float]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header in {path}")
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            rows.append({key: float(row[key]) for key in REQUIRED_COLUMNS})
    if len(rows) < 5:
        raise ValueError(f"Not enough rows in {path}")
    data = {key: np.array([row[key] for row in rows], dtype=np.float64) for key in REQUIRED_COLUMNS}
    return TrialData(**data)


def _hampel_filter(values: np.ndarray, window: int = 7, n_sigma: float = 3.0) -> np.ndarray:
    x = values.copy()
    if len(x) <= 2 * window:
        return x
    for i in range(window, len(x) - window):
        segment = x[i - window : i + window + 1]
        med = np.median(segment)
        mad = np.median(np.abs(segment - med))
        threshold = n_sigma * 1.4826 * max(mad, 1e-9)
        if abs(x[i] - med) > threshold:
            x[i] = med
    return x


def _smooth(values: np.ndarray, dt: float) -> np.ndarray:
    if signal is None:
        kernel = np.ones(5, dtype=np.float64) / 5.0
        return np.convolve(values, kernel, mode="same")
    nyquist = 0.5 / dt
    cutoff = min(0.8, 0.35 * nyquist)
    b, a = signal.butter(2, cutoff / nyquist, btype="low")
    return signal.filtfilt(b, a, values)


def _derivative(values: np.ndarray, dt: float) -> np.ndarray:
    if signal is not None and len(values) >= 9:
        window = 9 if len(values) >= 9 else len(values) - (1 - len(values) % 2)
        return signal.savgol_filter(values, window_length=window, polyorder=3, deriv=1, delta=dt)
    return np.gradient(values, dt)


def preprocess_trial(trial: TrialData, dt: float = 0.1) -> tuple[TrialData, dict[str, np.ndarray]]:
    t_uniform = np.arange(trial.t[0], trial.t[-1], dt)
    fields = {}
    for key in ("x", "y", "psi", "u", "v", "r", "delta", "n"):
        interpolated = np.interp(t_uniform, trial.t, getattr(trial, key))
        filtered = _hampel_filter(interpolated)
        fields[key] = _smooth(filtered, dt)
    processed = TrialData(t=t_uniform, **fields)
    deriv = {
        "du": _derivative(processed.u, dt),
        "dv": _derivative(processed.v, dt),
        "dr": _derivative(processed.r, dt),
    }
    return processed, deriv


def fit_nomoto(trials: Iterable[tuple[TrialData, dict[str, np.ndarray]]]) -> tuple[float, float]:
    lhs = []
    rhs = []
    for trial, deriv in trials:
        lhs.append(np.column_stack([-trial.r, trial.delta]))
        rhs.append(deriv["dr"])
    h = np.concatenate(lhs, axis=0)
    y = np.concatenate(rhs, axis=0)
    coeff, *_ = np.linalg.lstsq(h, y, rcond=None)
    a, b = coeff
    nomoto_t = max(0.1, 1.0 / max(a, 1e-6))
    nomoto_k = b * nomoto_t
    return float(nomoto_k), float(nomoto_t)


def fit_mmg_least_squares(
    trials: Iterable[tuple[TrialData, dict[str, np.ndarray]]],
    masses: tuple[float, float, float] = (35.0, 45.0, 18.0),
    l2_reg: float = 1e-4,
) -> MMGParameters:
    m_u, m_v, i_z = masses
    phi_x = []
    y_x = []
    phi_y = []
    y_y = []
    phi_n = []
    y_n = []
    for trial, deriv in trials:
        phi_x.append(np.column_stack([trial.u, np.abs(trial.u) * trial.u, trial.n * np.abs(trial.n)]))
        y_x.append(m_u * (deriv["du"] - trial.r * trial.v))
        phi_y.append(np.column_stack([trial.v, trial.r, np.abs(trial.v) * trial.v, trial.u**2 * trial.delta]))
        y_y.append(m_v * (deriv["dv"] + trial.r * trial.u))
        phi_n.append(np.column_stack([trial.v, trial.r, np.abs(trial.r) * trial.r, trial.u**2 * trial.delta]))
        y_n.append(i_z * deriv["dr"])
    x_mat = np.concatenate(phi_x, axis=0)
    x_tar = np.concatenate(y_x, axis=0)
    y_mat = np.concatenate(phi_y, axis=0)
    y_tar = np.concatenate(y_y, axis=0)
    n_mat = np.concatenate(phi_n, axis=0)
    n_tar = np.concatenate(y_n, axis=0)

    def solve(phi: np.ndarray, target: np.ndarray) -> np.ndarray:
        gram = phi.T @ phi + l2_reg * np.eye(phi.shape[1])
        return np.linalg.solve(gram, phi.T @ target)

    cx = solve(x_mat, x_tar)
    cy = solve(y_mat, y_tar)
    cn = solve(n_mat, n_tar)
    return MMGParameters(
        m_u=m_u,
        m_v=m_v,
        i_z=i_z,
        x_u=float(cx[0]),
        x_uu=float(cx[1]),
        x_n=float(cx[2]),
        y_v=float(cy[0]),
        y_r=float(cy[1]),
        y_vv=float(cy[2]),
        y_delta=float(cy[3]),
        n_v=float(cn[0]),
        n_r=float(cn[1]),
        n_rr=float(cn[2]),
        n_delta=float(cn[3]),
    )


def _rollout_error(trial: TrialData, params: MMGParameters, dt: float) -> float:
    dynamics = MMGDynamics(params=params)
    state = ShipState(
        x=float(trial.x[0]),
        y=float(trial.y[0]),
        psi=float(trial.psi[0]),
        u=float(trial.u[0]),
        v=float(trial.v[0]),
        r=float(trial.r[0]),
        delta=float(trial.delta[0]),
        n=float(trial.n[0]),
    )
    sq_error = []
    for i in range(1, len(trial.t)):
        dot_delta = (trial.delta[i] - trial.delta[i - 1]) / dt
        dot_n = (trial.n[i] - trial.n[i - 1]) / dt
        state = dynamics.step(state, dot_delta_cmd=dot_delta, dot_n_cmd=dot_n, dt=dt, disturbance_body=(0.0, 0.0))
        err = (state.psi - trial.psi[i]) ** 2 + (state.r - trial.r[i]) ** 2
        sq_error.append(err)
    return float(np.mean(sq_error))


def refine_mmg_multiple_shooting(
    trials: Iterable[TrialData],
    init_params: MMGParameters,
    dt: float = 0.1,
) -> MMGParameters:
    if optimize is None:  # pragma: no cover
        return init_params
    names = ["x_u", "x_uu", "x_n", "y_v", "y_r", "y_vv", "y_delta", "n_v", "n_r", "n_rr", "n_delta"]
    x0 = np.array([getattr(init_params, name) for name in names], dtype=np.float64)
    trial_list = list(trials)

    def loss(theta: np.ndarray) -> float:
        params = MMGParameters(**asdict(init_params))
        for i, name in enumerate(names):
            setattr(params, name, float(theta[i]))
        rollout = np.mean([_rollout_error(trial, params, dt) for trial in trial_list])
        reg = 1e-3 * np.mean((theta - x0) ** 2)
        return float(rollout + reg)

    result = optimize.minimize(loss, x0, method="L-BFGS-B", options={"maxiter": 200})
    params = MMGParameters(**asdict(init_params))
    for i, name in enumerate(names):
        setattr(params, name, float(result.x[i]))
    return params


def validate_parameters(trials: Iterable[TrialData], params: MMGParameters, dt: float = 0.1) -> dict[str, float]:
    psi_errors = []
    r_errors = []
    endpoint_errors = []
    for trial in trials:
        dynamics = MMGDynamics(params=params)
        state = ShipState(
            x=float(trial.x[0]),
            y=float(trial.y[0]),
            psi=float(trial.psi[0]),
            u=float(trial.u[0]),
            v=float(trial.v[0]),
            r=float(trial.r[0]),
            delta=float(trial.delta[0]),
            n=float(trial.n[0]),
        )
        pred_psi = []
        pred_r = []
        for i in range(1, len(trial.t)):
            dot_delta = (trial.delta[i] - trial.delta[i - 1]) / dt
            dot_n = (trial.n[i] - trial.n[i - 1]) / dt
            state = dynamics.step(state, dot_delta_cmd=dot_delta, dot_n_cmd=dot_n, dt=dt, disturbance_body=(0.0, 0.0))
            pred_psi.append(state.psi)
            pred_r.append(state.r)
        psi_true = trial.psi[1 : len(pred_psi) + 1]
        r_true = trial.r[1 : len(pred_r) + 1]
        psi_errors.append(np.sqrt(np.mean((np.array(pred_psi) - psi_true) ** 2)))
        r_errors.append(np.sqrt(np.mean((np.array(pred_r) - r_true) ** 2)))
        endpoint_errors.append(np.sqrt((state.x - trial.x[-1]) ** 2 + (state.y - trial.y[-1]) ** 2))
    return {
        "rmse_psi_rad": float(np.mean(psi_errors)),
        "rmse_psi_deg": float(np.rad2deg(np.mean(psi_errors))),
        "rmse_r": float(np.mean(r_errors)),
        "endpoint_error_m": float(np.mean(endpoint_errors)),
    }


def save_identification_result(path: str | Path, result: IdentificationResult) -> None:
    payload = {
        "nomoto_k": result.nomoto_k,
        "nomoto_t": result.nomoto_t,
        "mmg_params": asdict(result.mmg_params),
        "validation": result.validation,
    }
    Path(path).write_text(json.dumps(payload, indent=2))
