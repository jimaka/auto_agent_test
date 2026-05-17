cmake_minimum_required(VERSION 3.0.2)
project(vessel_control)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  vessel_msgs
)

find_package(Eigen3 QUIET)
if(NOT EIGEN3_FOUND)
  message(WARNING "Eigen3 not found; install libeigen3-dev for MPC build")
endif()

# Optional: OSQP + osqp-eigen when available
# find_package(osqp REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp std_msgs vessel_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(vessel_control_node
  src/ros_node.cpp
  src/state_assembly.cpp
  src/koopman_lift.cpp
  src/mpc_osqp.cpp
  src/tube_constraints.cpp
  src/trajectory_tracker.cpp
)

target_link_libraries(vessel_control_node
  ${catkin_LIBRARIES}
)

add_dependencies(vessel_control_node ${catkin_EXPORTED_TARGETS} ${vessel_msgs_EXPORTED_TARGETS})

install(TARGETS vessel_control_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(DIRECTORY launch config model_registry/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  USE_SOURCE_PERMISSIONS
)
