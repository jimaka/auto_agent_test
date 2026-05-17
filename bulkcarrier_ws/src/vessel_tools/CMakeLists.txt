cmake_minimum_required(VERSION 3.0.2)
project(vessel_tools)

find_package(catkin REQUIRED)
catkin_python_setup()
catkin_package()

catkin_install_python(PROGRAMS
  scripts/bag2dataset.py
  scripts/time_sync.py
  scripts/bench_latency.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test)
endif()
