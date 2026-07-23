cmake_minimum_required(VERSION 3.0.2)
project(vessel_msgs)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  std_msgs
  geometry_msgs
)

add_message_files(
  FILES
  VesselState.msg
  ControlCmd.msg
  ControlStatus.msg
  TrajectoryPoint.msg
  TrajectoryRef.msg
  Wind.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs geometry_msgs
)
