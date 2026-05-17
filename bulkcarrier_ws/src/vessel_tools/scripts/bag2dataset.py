#!/usr/bin/env python3
"""Convert a single ROS bag to a resampled dataset (NPZ/CSV)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from source tree without catkin devel
_PKG_SRC = Path(__file__).resolve().parents[1] / "src"
if _PKG_SRC.is_dir() and str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

from vessel_tools.pipeline import bag_to_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="ROS bag -> processed dataset")
    parser.add_argument("--bag", required=True, help="Input .bag path")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--ts", type=float, default=0.25, help="Resample period [s]")
    parser.add_argument("--ins-topic", default="/ins/state")
    parser.add_argument("--rudder-topic", default="/sensors/rudder_deg")
    parser.add_argument("--rpm-topic", default="/sensors/shaft_rpm")
    parser.add_argument("--wind-topic", default="/sensors/wind")
    parser.add_argument("--no-wind", action="store_true")
    parser.add_argument(
        "--rudder-in-degrees",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Convert rudder channel from degrees to radians (default: true)",
    )
    parser.add_argument(
        "--backend",
        choices=["rosbag", "rosbags", "auto"],
        default="auto",
        help="Bag reader backend",
    )
    args = parser.parse_args()

    topics = {
        "ins": args.ins_topic,
        "rudder_deg": args.rudder_topic,
        "shaft_rpm": args.rpm_topic,
    }
    if not args.no_wind:
        topics["wind"] = args.wind_topic

    backend = None if args.backend == "auto" else args.backend
    out = bag_to_dataset(
        bag_path=args.bag,
        out_dir=args.out,
        Ts=args.ts,
        topics=topics,
        rudder_in_degrees=args.rudder_in_degrees,
        backend=backend,
    )
    print(f"[vessel_tools] Wrote dataset to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
