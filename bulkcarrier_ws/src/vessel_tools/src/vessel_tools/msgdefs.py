"""ROS1 message definitions for rosbags deserialization (no catkin required)."""

from __future__ import annotations

from rosbags.typesys import Stores, get_types_from_msg, get_typestore

HEADER = "std_msgs/Header"
TIME = "time"

STD_FLOAT64 = """
float64 data
"""

VESSEL_STATE = """
std_msgs/Header header
float64 x
float64 y
float64 psi
float64 u
float64 v
float64 r
"""

WIND = """
std_msgs/Header header
float64 speed_mps
float64 direction_rad
"""


def build_typestore():
    store = get_typestore(Stores.ROS1_NOETIC)
    add = {}
    add.update(get_types_from_msg(STD_FLOAT64, "std_msgs/msg/Float64"))
    add.update(get_types_from_msg(VESSEL_STATE, "vessel_msgs/msg/VesselState"))
    add.update(get_types_from_msg(WIND, "vessel_msgs/msg/Wind"))
    store.register(add)
    return store
