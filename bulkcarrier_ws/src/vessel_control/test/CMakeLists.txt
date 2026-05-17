cmake_minimum_required(VERSION 3.14)
project(vessel_control_standalone_test)

set(CMAKE_CXX_STANDARD 14)

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/vessel_deps.cmake)

add_executable(mpc_standalone_test
  mpc_standalone_test.cpp
  ../src/model_loader.cpp
  ../src/koopman_lift.cpp
  ../src/mpc_osqp.cpp
  ../src/osqp_interface.cpp
  ../src/tube_constraints.cpp
)

target_include_directories(mpc_standalone_test PRIVATE ../include)

if(VESSEL_USE_OSQP AND TARGET ${VESSEL_OSQP_TARGET})
  target_compile_definitions(mpc_standalone_test PRIVATE VESSEL_USE_OSQP)
  target_link_libraries(mpc_standalone_test ${VESSEL_OSQP_TARGET})
  target_include_directories(mpc_standalone_test PRIVATE ${osqp_SOURCE_DIR}/include)
endif()

if(VESSEL_USE_ONNXRUNTIME AND TARGET ${VESSEL_ONNXRUNTIME_TARGET})
  target_compile_definitions(mpc_standalone_test PRIVATE VESSEL_USE_ONNXRUNTIME)
  target_link_libraries(mpc_standalone_test ${VESSEL_ONNXRUNTIME_TARGET})
endif()

add_executable(rate_constraint_test
  rate_constraint_test.cpp
  ../src/model_loader.cpp
  ../src/koopman_lift.cpp
  ../src/mpc_osqp.cpp
  ../src/osqp_interface.cpp
  ../src/tube_constraints.cpp
)
target_include_directories(rate_constraint_test PRIVATE ../include)
if(VESSEL_USE_OSQP AND TARGET ${VESSEL_OSQP_TARGET})
  target_compile_definitions(rate_constraint_test PRIVATE VESSEL_USE_OSQP)
  target_link_libraries(rate_constraint_test ${VESSEL_OSQP_TARGET})
  target_include_directories(rate_constraint_test PRIVATE ${osqp_SOURCE_DIR}/include)
endif()
if(VESSEL_USE_ONNXRUNTIME AND TARGET ${VESSEL_ONNXRUNTIME_TARGET})
  target_compile_definitions(rate_constraint_test PRIVATE VESSEL_USE_ONNXRUNTIME)
  target_link_libraries(rate_constraint_test ${VESSEL_ONNXRUNTIME_TARGET})
endif()
