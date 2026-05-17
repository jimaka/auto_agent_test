"""End-to-end manifest-driven dataset pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from vessel_tools.bag_reader import merge_bag_streams, read_bag
from vessel_tools.dataset_io import export_dataset
from vessel_tools.manifest import Manifest, load_manifest
from vessel_tools.sync import chronological_splits, sync_streams


def process_manifest(
    manifest_path: str | Path,
    repo_root: Optional[str | Path] = None,
    backend: Optional[str] = None,
) -> Path:
    manifest = load_manifest(manifest_path, repo_root=repo_root)
    return process_from_manifest(manifest, backend=backend)


def process_from_manifest(manifest: Manifest, backend: Optional[str] = None) -> Path:
    bag_parts = []
    for src in manifest.sources:
        if not src.path.is_file():
            raise FileNotFoundError(f"Source bag missing: {src.path}")
        bag_parts.append(
            read_bag(src.path, manifest.topics, source_id=src.source_id or src.path.stem, backend=backend)
        )

    merged = merge_bag_streams(bag_parts)
    synced = sync_streams(merged, manifest.Ts, manifest.processing)
    splits = chronological_splits(len(synced.t), manifest.splits)
    return export_dataset(synced, manifest, splits)


def bag_to_dataset(
    bag_path: str | Path,
    out_dir: str | Path,
    Ts: float = 0.25,
    topics: Optional[dict] = None,
    rudder_in_degrees: bool = True,
    backend: Optional[str] = None,
) -> Path:
    """Single-bag convenience API (no manifest file)."""
    from vessel_tools.manifest import Manifest, OutputSpec, ProcessingSpec, SourceSpec, TopicMap

    bag_path = Path(bag_path)
    out_dir = Path(out_dir)
    topics = topics or {
        "ins": "/ins/state",
        "rudder_deg": "/sensors/rudder_deg",
        "shaft_rpm": "/sensors/shaft_rpm",
        "wind": "/sensors/wind",
    }
    tm = TopicMap(
        ins=topics["ins"],
        rudder_deg=topics["rudder_deg"],
        shaft_rpm=topics["shaft_rpm"],
        wind=topics.get("wind"),
    )
    manifest = Manifest(
        dataset_id=out_dir.name,
        ship_id="unknown",
        Ts=Ts,
        topics=tm,
        sources=[SourceSpec(type="rosbag", path=bag_path, source_id=bag_path.stem)],
        output=OutputSpec(root=out_dir, format="npz"),
        processing=ProcessingSpec(rudder_in_degrees=rudder_in_degrees),
        splits={"train": 1.0},
    )
    streams = read_bag(bag_path, tm, backend=backend)
    synced = sync_streams(streams, Ts, manifest.processing)
    splits = chronological_splits(len(synced.t), manifest.splits)
    return export_dataset(synced, manifest, splits)
