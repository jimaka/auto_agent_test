cmake_minimum_required(VERSION 3.0.2)
project(vessel_baseline)

find_package(catkin REQUIRED COMPONENTS roscpp vessel_msgs)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp vessel_msgs
)

include_directories(include ${catkin_INCLUDE_DIRS})

add_executable(vessel_baseline_node
  src/baseline_node.cpp
  src/nomoto_mpc.cpp
)
target_link_libraries(vessel_baseline_node ${catkin_LIBRARIES})
add_dependencies(vessel_baseline_node ${catkin_EXPORTED_TARGETS} ${vessel_msgs_EXPORTED_TARGETS})

install(TARGETS vessel_baseline_node RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY launch config/ DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
