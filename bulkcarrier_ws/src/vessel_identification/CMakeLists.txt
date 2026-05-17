cmake_minimum_required(VERSION 3.0.2)
project(vessel_identification)

find_package(catkin REQUIRED)
catkin_python_setup()
catkin_package()

catkin_install_python(PROGRAMS
  scripts/train.py
  scripts/validate_horizon.py
  scripts/export_model.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test)
endif()
