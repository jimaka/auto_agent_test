from __future__ import annotations

import json
from pathlib import Path


def _as_pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _gate_line(gates: dict[str, object]) -> str:
    parts = []
    for gate in ("M1", "M2", "M3", "M4"):
        gate_payload = gates.get(gate, {})
        passed = bool(gate_payload.get(f"{gate.lower()}_passed", False))
        parts.append(f"{gate}={'PASS' if passed else 'FAIL'}")
    return ", ".join(parts)


def write_iteration_report(
    output_path: Path,
    iteration_idx: int,
    run_results: list[dict[str, object]],
    best_record: dict[str, object] | None,
    elapsed_s: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_runs = sorted(run_results, key=lambda item: float(item.get("score", -1e9)), reverse=True)

    lines = [
        f"# Iteration {iteration_idx:03d} Optimization Report",
        "",
        f"- Runs in batch: {len(run_results)}",
        f"- Elapsed: {elapsed_s:.1f}s",
        "",
        "## Batch Results",
        "",
    ]
    for idx, result in enumerate(sorted_runs, start=1):
        metrics = result.get("metrics", {})
        gates = result.get("gates", {})
        lines.extend(
            [
                f"### {idx}. `{result.get('run_tag', 'unknown')}`",
                f"- Score: `{float(result.get('score', 0.0)):.3f}`",
                f"- Gates: {_gate_line(gates if isinstance(gates, dict) else {})}",
                f"- Success rate: `{_as_pct(float(metrics.get('success_rate', 0.0)))}`",
                f"- Collision rate: `{_as_pct(float(metrics.get('collision_rate', 0.0)))}`",
                f"- Out-of-channel rate: `{_as_pct(float(metrics.get('out_of_channel_rate', 0.0)))}`",
                f"- Takeover recommendation rate: `{_as_pct(float(metrics.get('takeover_recommendation_rate', 0.0)))}`",
                f"- Report: `{result.get('report_path', '')}`",
                "",
            ]
        )

    lines.extend(["## Best So Far", ""])
    if best_record is None:
        lines.append("- None yet.")
    else:
        lines.extend(
            [
                f"- Run tag: `{best_record.get('run_tag', 'unknown')}`",
                f"- Score: `{float(best_record.get('score', 0.0)):.3f}`",
                f"- Gates: {_gate_line(best_record.get('gates', {}))}",
                f"- Report: `{best_record.get('report_path', '')}`",
            ]
        )
    lines.append("")
    output_path.write_text("\n".join(lines))


def write_final_report(
    output_path: Path,
    summary: dict[str, object],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best = summary.get("best_run", {})
    completed_runs = summary.get("completed_runs", [])
    lines = [
        "# Ship3DOF Autotune Final Summary",
        "",
        f"- Total runs: {summary.get('total_runs', 0)}",
        f"- Completed runs: {len(completed_runs) if isinstance(completed_runs, list) else 0}",
        f"- Time budget hours: {summary.get('time_budget_hours', 0)}",
        f"- Actual elapsed seconds: {float(summary.get('elapsed_s', 0.0)):.1f}",
        "",
        "## Best Configuration",
        "",
        f"- Run tag: `{best.get('run_tag', 'n/a')}`",
        f"- Score: `{float(best.get('score', 0.0)):.3f}`",
        f"- Gates: {_gate_line(best.get('gates', {}))}",
        f"- Success rate: `{_as_pct(float(best.get('metrics', {}).get('success_rate', 0.0)) if isinstance(best, dict) else 0.0)}`",
        f"- Takeover recommendation rate: `{_as_pct(float(best.get('metrics', {}).get('takeover_recommendation_rate', 0.0)) if isinstance(best, dict) else 0.0)}`",
        f"- Report path: `{best.get('report_path', 'n/a')}`",
        "",
        "## Full Summary JSON",
        "",
        "```json",
        json.dumps(summary, indent=2),
        "```",
    ]
    output_path.write_text("\n".join(lines))
