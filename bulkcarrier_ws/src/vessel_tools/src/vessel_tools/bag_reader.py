"""Read ROS1 bags into SampleStream objects."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np

from vessel_tools.manifest import TopicMap
from vessel_tools.streams import BagStreams, SampleStream

# Optional backends
_ROSBAG = None
_ROSBAGS = None


def _load_rosbag():
    global _ROSBAG
    if _ROSBAG is None:
        import rosbag  # type: ignore

        _ROSBAG = rosbag
    return _ROSBAG


def _load_rosbags():
    global _ROSBAGS
    if _ROSBAGS is None:
        from rosbags.rosbag1 import Reader as Rosbag1Reader

        _ROSBAGS = Rosbag1Reader
    return _ROSBAGS


def _stamp_to_sec(stamp) -> float:
    if hasattr(stamp, "to_sec"):
        return float(stamp.to_sec())
    if isinstance(stamp, (int, float)):
        return float(stamp)
    return float(stamp.secs) + float(stamp.nsecs) * 1e-9


def read_bag(
    bag_path: Path,
    topics: TopicMap,
    source_id: str = "",
    backend: Optional[str] = None,
) -> BagStreams:
    bag_path = Path(bag_path)
    if not bag_path.is_file():
        raise FileNotFoundError(f"Bag not found: {bag_path}")

    if backend is None:
        backend = "rosbag" if _rosbag_available() else "rosbags"

    if backend == "rosbag":
        return _read_rosbag(bag_path, topics, source_id)
    if backend == "rosbags":
        return _read_rosbags(bag_path, topics, source_id)
    raise ValueError(f"Unknown backend: {backend}")


def _rosbag_available() -> bool:
    try:
        import rosbag  # noqa: F401

        return True
    except ImportError:
        return False


def _read_rosbag(bag_path: Path, topics: TopicMap, source_id: str) -> BagStreams:
    rosbag = _load_rosbag()
    ins_t, ins_v = [], []
    rud_t, rud_v = [], []
    rpm_t, rpm_v = [], []
    wind_t, wind_v = [], []

    with rosbag.Bag(str(bag_path), "r") as bag:
        topic_map = {
            topics.ins: ("ins", ins_t, ins_v, _parse_vessel_state_ros),
            topics.rudder_deg: ("rudder", rud_t, rud_v, _parse_float64_ros),
            topics.shaft_rpm: ("rpm", rpm_t, rpm_v, _parse_float64_ros),
        }
        if topics.wind:
            topic_map[topics.wind] = ("wind", wind_t, wind_v, _parse_wind_ros)

        for topic, msg, t in bag.read_messages(topics=list(topic_map.keys())):
            _, tlist, vlist, parser = topic_map[topic]
            tlist.append(t.to_sec() if hasattr(t, "to_sec") else float(t))
            vlist.append(parser(msg))

    wind = None
    if topics.wind and wind_t:
        wind = SampleStream(np.array(wind_t), np.array(wind_v), name="wind")

    return BagStreams(
        ins=SampleStream(np.array(ins_t), np.array(ins_v), name="ins"),
        rudder_deg=SampleStream(np.array(rud_t), np.array(rud_v), name="rudder_deg"),
        shaft_rpm=SampleStream(np.array(rpm_t), np.array(rpm_v), name="shaft_rpm"),
        wind=wind,
        source_id=source_id,
    )


def _read_rosbags(bag_path: Path, topics: TopicMap, source_id: str) -> BagStreams:
    from vessel_tools.msgdefs import build_typestore

    Reader = _load_rosbags()
    store = build_typestore()

    ins_t, ins_v = [], []
    rud_t, rud_v = [], []
    rpm_t, rpm_v = [], []
    wind_t, wind_v = [], []

    wanted = {
        topics.ins: ("ins", ins_t, ins_v, "vessel_msgs/msg/VesselState"),
        topics.rudder_deg: ("rudder", rud_t, rud_v, "std_msgs/msg/Float64"),
        topics.shaft_rpm: ("rpm", rpm_t, rpm_v, "std_msgs/msg/Float64"),
    }
    if topics.wind:
        wanted[topics.wind] = ("wind", wind_t, wind_v, "vessel_msgs/msg/Wind")

    with Reader(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic in wanted]
        for conn, t, raw in reader.messages(connections=connections):
            _, tlist, vlist, typename = wanted[conn.topic]
            msg = store.deserialize_ros1(raw, typename)
            tlist.append(t * 1e-9)
            if typename.endswith("VesselState"):
                vlist.append(_parse_vessel_state_msg(msg))
            elif typename.endswith("Wind"):
                vlist.append(_parse_wind_msg(msg))
            else:
                vlist.append([float(msg.data)])

    wind = None
    if topics.wind and wind_t:
        wind = SampleStream(np.array(wind_t), np.array(wind_v), name="wind")

    return BagStreams(
        ins=SampleStream(np.array(ins_t), np.array(ins_v), name="ins"),
        rudder_deg=SampleStream(np.array(rud_t), np.array(rud_v), name="rudder_deg"),
        shaft_rpm=SampleStream(np.array(rpm_t), np.array(rpm_v), name="shaft_rpm"),
        wind=wind,
        source_id=source_id,
    )


def _parse_vessel_state_ros(msg) -> list:
    return [float(msg.x), float(msg.y), float(msg.psi), float(msg.u), float(msg.v), float(msg.r)]


def _parse_wind_ros(msg) -> list:
    return [float(msg.speed_mps), float(msg.direction_rad)]


def _parse_float64_ros(msg) -> list:
    return [float(msg.data)]


def _parse_vessel_state_msg(msg) -> list:
    return [float(msg.x), float(msg.y), float(msg.psi), float(msg.u), float(msg.v), float(msg.r)]


def _parse_wind_msg(msg) -> list:
    return [float(msg.speed_mps), float(msg.direction_rad)]


def merge_bag_streams(streams_list: list[BagStreams]) -> BagStreams:
    """Concatenate multiple bag extractions sorted by time."""
    if not streams_list:
        raise ValueError("merge_bag_streams: empty list")
    if len(streams_list) == 1:
        s = streams_list[0]
        for st in s.all_streams():
            st.sort_inplace()
            st.dedupe_inplace()
        return s

    def _merge(name: str, getter: Callable[[BagStreams], SampleStream]) -> SampleStream:
        times, vals = [], []
        for s in streams_list:
            st = getter(s)
            times.append(st.times)
            vals.append(st.values)
        return SampleStream(np.concatenate(times), np.concatenate(vals), name=name)

    wind_parts = [s.wind for s in streams_list if s.wind is not None]
    wind = None
    if wind_parts:
        times = np.concatenate([w.times for w in wind_parts])
        vals = np.concatenate([w.values for w in wind_parts])
        wind = SampleStream(times, vals, name="wind")

    merged = BagStreams(
        ins=_merge("ins", lambda s: s.ins),
        rudder_deg=_merge("rudder_deg", lambda s: s.rudder_deg),
        shaft_rpm=_merge("shaft_rpm", lambda s: s.shaft_rpm),
        wind=wind,
        source_id="merged",
    )
    for st in merged.all_streams():
        st.sort_inplace()
        st.dedupe_inplace()
    return merged
