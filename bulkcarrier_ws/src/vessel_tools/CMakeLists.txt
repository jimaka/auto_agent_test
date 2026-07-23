cmake_minimum_required(VERSION 3.0.2)
project(vessel_tools)

find_package(catkin REQUIRED)
catkin_package()

catkin_install_python(PROGRAMS
  scripts/bag2dataset.py
  scripts/time_sync.py
  scripts/bench_latency.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
