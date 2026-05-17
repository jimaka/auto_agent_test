"""Load and validate dataset manifest YAML files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class TopicMap:
    ins: str
    rudder_deg: str
    shaft_rpm: str
    wind: Optional[str] = None


@dataclass
class SourceSpec:
    type: str
    path: Path
    source_id: Optional[str] = None


@dataclass
class OutputSpec:
    root: Path
    format: str = "npz"


@dataclass
class ProcessingSpec:
    max_gap_sec: float = 1.0
    rudder_in_degrees: bool = True


@dataclass
class Manifest:
    dataset_id: str
    ship_id: str
    Ts: float
    topics: TopicMap
    sources: List[SourceSpec]
    output: OutputSpec
    processing: ProcessingSpec = field(default_factory=ProcessingSpec)
    splits: Dict[str, float] = field(default_factory=lambda: {"train": 0.7, "val": 0.15, "test": 0.15})
    manifest_path: Optional[Path] = None

    @property
    def repo_root(self) -> Path:
        if self.manifest_path is not None:
            return self.manifest_path.parent.parent.parent
        return Path.cwd()


def _resolve(path: Path, base: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p.resolve()
    return (base / p).resolve()


def load_manifest(path: str | Path, repo_root: Optional[str | Path] = None) -> Manifest:
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    base = Path(repo_root).resolve() if repo_root else manifest_path.parent
    # data/manifests/foo.yaml -> repo root often two levels up from manifests
    if repo_root is None and (manifest_path.parent.name == "manifests"):
        base = manifest_path.parent.parent.parent

    topics_raw = raw.get("topics", {})
    topics = TopicMap(
        ins=topics_raw["ins"],
        rudder_deg=topics_raw["rudder_deg"],
        shaft_rpm=topics_raw["shaft_rpm"],
        wind=topics_raw.get("wind"),
    )

    sources: List[SourceSpec] = []
    for i, src in enumerate(raw.get("sources", [])):
        if src.get("type", "rosbag") != "rosbag":
            raise ValueError(f"Unsupported source type: {src.get('type')}")
        sources.append(
            SourceSpec(
                type="rosbag",
                path=_resolve(src["path"], base),
                source_id=src.get("id", f"source_{i}"),
            )
        )

    out_raw = raw.get("output", {})
    if "root" in out_raw:
        out_root = _resolve(out_raw["root"], base)
    else:
        out_root = _resolve(f"data/processed/{raw['dataset_id']}", base)

    proc_raw = raw.get("processing", {})
    processing = ProcessingSpec(
        max_gap_sec=float(proc_raw.get("max_gap_sec", 1.0)),
        rudder_in_degrees=bool(proc_raw.get("rudder_in_degrees", True)),
    )

    splits = raw.get("splits", {"train": 0.7, "val": 0.15, "test": 0.15})
    _validate_splits(splits)

    m = Manifest(
        dataset_id=str(raw["dataset_id"]),
        ship_id=str(raw.get("ship_id", "unknown")),
        Ts=float(raw.get("Ts", 0.25)),
        topics=topics,
        sources=sources,
        output=OutputSpec(root=out_root, format=str(out_raw.get("format", "npz")).lower()),
        processing=processing,
        splits={k: float(v) for k, v in splits.items()},
        manifest_path=manifest_path,
    )
    return m


def _validate_splits(splits: Dict[str, float]) -> None:
    if not splits:
        raise ValueError("splits must not be empty")
    total = sum(splits.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"splits must sum to 1.0, got {total}")
