# Optional dependencies for vessel_control

option(VESSEL_USE_OSQP "Build with OSQP solver" ON)
option(VESSEL_USE_ONNXRUNTIME "Build with ONNX Runtime encoder" ON)

if(VESSEL_USE_OSQP)
  include(FetchContent)
  set(OSQP_BUILD_DEMO_EXE OFF CACHE BOOL "" FORCE)
  set(OSQP_BUILD_UNITTESTS OFF CACHE BOOL "" FORCE)
  set(OSQP_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    osqp
    GIT_REPOSITORY https://github.com/osqp/osqp.git
    GIT_TAG v0.6.3
  )
  FetchContent_MakeAvailable(osqp)
  set(VESSEL_OSQP_TARGET osqpstatic)
endif()

if(VESSEL_USE_ONNXRUNTIME)
  set(ONNXRUNTIME_VERSION "1.17.1" CACHE STRING "ONNX Runtime version")
  set(ONNXRUNTIME_DIR "${CMAKE_BINARY_DIR}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}")
  if(NOT EXISTS "${ONNXRUNTIME_DIR}/include/onnxruntime_cxx_api.h")
    set(ONNX_TGZ "${CMAKE_BINARY_DIR}/onnxruntime.tgz")
    message(STATUS "Downloading ONNX Runtime ${ONNXRUNTIME_VERSION} ...")
    file(DOWNLOAD
      "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz"
      "${ONNX_TGZ}"
      SHOW_PROGRESS
      TLS_VERIFY ON
    )
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E tar xzf "${ONNX_TGZ}"
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
  endif()
  if(EXISTS "${ONNXRUNTIME_DIR}/include/onnxruntime_cxx_api.h")
    add_library(onnxruntime SHARED IMPORTED GLOBAL)
    set_target_properties(onnxruntime PROPERTIES
      IMPORTED_LOCATION "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so"
      INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_DIR}/include"
    )
    set(VESSEL_ONNXRUNTIME_TARGET onnxruntime)
  else()
    message(WARNING "ONNX Runtime not found after download; disabling VESSEL_USE_ONNXRUNTIME")
    set(VESSEL_USE_ONNXRUNTIME OFF)
  endif()
endif()
