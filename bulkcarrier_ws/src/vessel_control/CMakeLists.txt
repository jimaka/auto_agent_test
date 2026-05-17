cmake_minimum_required(VERSION 3.14)
project(vessel_control)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  vessel_msgs
)

find_package(Eigen3 QUIET)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/vessel_deps.cmake)

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
  src/model_loader.cpp
  src/koopman_lift.cpp
  src/mpc_osqp.cpp
  src/osqp_interface.cpp
  src/tube_constraints.cpp
  src/trajectory_tracker.cpp
)

target_compile_features(vessel_control_node PRIVATE cxx_std_14)
target_compile_definitions(vessel_control_node PRIVATE VESSEL_HAS_ROS=1)

target_link_libraries(vessel_control_node
  ${catkin_LIBRARIES}
)

if(VESSEL_USE_OSQP AND TARGET ${VESSEL_OSQP_TARGET})
  target_compile_definitions(vessel_control_node PRIVATE VESSEL_USE_OSQP)
  target_link_libraries(vessel_control_node ${VESSEL_OSQP_TARGET})
  target_include_directories(vessel_control_node PRIVATE ${osqp_SOURCE_DIR}/include)
endif()

if(VESSEL_USE_ONNXRUNTIME AND TARGET ${VESSEL_ONNXRUNTIME_TARGET})
  target_compile_definitions(vessel_control_node PRIVATE VESSEL_USE_ONNXRUNTIME)
  target_link_libraries(vessel_control_node ${VESSEL_ONNXRUNTIME_TARGET})
endif()

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
