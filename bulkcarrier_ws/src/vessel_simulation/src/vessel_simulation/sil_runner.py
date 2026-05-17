"""MMG software-in-the-loop closed-loop benchmark runner."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol

import numpy as np
import yaml

from vessel_simulation.metrics import BenchmarkMetrics, aggregate_metrics, compute_sample
from vessel_simulation.mmg3dof import ControlInput, MMG3DOF, VesselState
from vessel_simulation.trajectory import TrajectoryRef, make_scenario


class Controller(Protocol):
    def compute(self, state: VesselState, u_prev: ControlInput, x_refs: List[VesselState]) -> ControlInput:
        ...


@dataclass
class SilConfig:
    dt_sim: float = 0.05
    Ts_control: float = 0.25
    duration_s: float = 600.0
    u0: float = 5.0


@dataclass
class SilResult:
    controller: str
    scenario: str
    metrics: BenchmarkMetrics
    passed: bool
    log_path: Optional[Path] = None

    def to_dict(self) -> Dict:
        return {
            "controller": self.controller,
            "scenario": self.scenario,
            "metrics": self.metrics.to_dict(),
            "passed": bool(self.passed),
            "log_path": str(self.log_path) if self.log_path else None,
        }


def load_sil_config(config_path: Path) -> SilConfig:
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    sil = raw.get("sil", {})
    mmg = raw.get("mmg", {})
    return SilConfig(
        dt_sim=float(mmg.get("dt", 0.05)),
        Ts_control=float(sil.get("control_Ts", 0.25)),
        duration_s=float(sil.get("default_duration_s", 600.0)),
        u0=float(sil.get("u0_mps", 5.0)),
    )


def run_sil(
    plant: MMG3DOF,
    traj: TrajectoryRef,
    controller: Controller,
    cfg: SilConfig,
    controller_name: str,
    out_dir: Optional[Path] = None,
    duration_s: Optional[float] = None,
) -> SilResult:
    duration = duration_s or cfg.duration_s
    t = 0.0
    ref0 = traj.state_at(0.0)
    state = VesselState(ref0.x, ref0.y, ref0.psi, cfg.u0, 0.0, 0.0)
    u_cmd = ControlInput(0.0, 60.0)
    u_applied = ControlInput(0.0, 60.0)

    samples = []
    log_rows: List[Dict] = []
    next_ctrl = 0.0
    horizon_steps = 0

    if hasattr(controller, "N"):
        horizon_steps = int(controller.N)
    elif hasattr(controller, "model_dir"):
        horizon_steps = 10

    while t < duration:
        ref = traj.state_at(t)
        if t >= next_ctrl - 1e-9:
            refs = [traj.state_at(t + k * cfg.Ts_control) for k in range(max(horizon_steps, 1))]
            u_cmd = controller.compute(state, u_applied, refs)
            next_ctrl += cfg.Ts_control

        # actuator rate limit (physical)
        du_delta = np.deg2rad(2.0) * cfg.Ts_control
        du_n = 5.0 * cfg.Ts_control
        delta = np.clip(
            u_cmd.delta_rad,
            u_applied.delta_rad - du_delta,
            u_applied.delta_rad + du_delta,
        )
        rpm = np.clip(u_cmd.rpm, u_applied.rpm - du_n, u_applied.rpm + du_n)
        u_applied = ControlInput(delta_rad=float(delta), rpm=float(rpm))

        samples.append(compute_sample(state, ref, t))
        log_rows.append(
            {
                "t": t,
                "x": state.x,
                "y": state.y,
                "psi": state.psi,
                "u": state.u,
                "delta": u_applied.delta_rad,
                "rpm": u_applied.rpm,
            }
        )

        state = plant.step(state, u_applied)
        t += plant.dt

    metrics = aggregate_metrics(samples)
    passed = _check_pass(metrics)

    log_path = None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / f"log_{controller_name}.json"
        with log_path.open("w", encoding="utf-8") as f:
            json.dump({"samples": [s.__dict__ for s in samples], "log": log_rows}, f, indent=2)

    return SilResult(controller_name, "scenario", metrics, passed, log_path)


def _check_pass(m: BenchmarkMetrics, cross_lim: float = 15.0, psi_lim: float = 0.5, u_lim: float = 2.0) -> bool:
    return (
        m.e_cross_rms < cross_lim
        and m.e_psi_rms < psi_lim
        and m.e_u_rms < u_lim
        and np.isfinite(m.e_cross_rms)
    )


def run_benchmark_suite(
    config_path: Path,
    model_dir: Path,
    scenario: str,
    duration_s: float,
    out_dir: Path,
) -> Dict:
    cfg = load_sil_config(config_path)
    plant = MMG3DOF.from_yaml(str(config_path))
    traj = make_scenario(scenario, u_ref=cfg.u0, duration=duration_s)

    from vessel_simulation.controllers.koopman_mpc import KoopmanMpcController
    from vessel_simulation.controllers.nomoto import NomotoController

    results = []
    for name, ctrl in [
        ("koopman_mpc", KoopmanMpcController(model_dir)),
        ("nomoto", NomotoController()),
    ]:
        r = run_sil(plant, traj, ctrl, cfg, name, out_dir / name, duration_s=duration_s)
        r.scenario = scenario
        results.append(r)

    report = {
        "scenario": scenario,
        "duration_s": duration_s,
        "model_dir": str(model_dir),
        "results": [r.to_dict() for r in results],
        "koopman_passed": bool(results[0].passed) if results else False,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "benchmark_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report
