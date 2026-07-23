#pragma once

#include <iostream>

#if defined(VESSEL_HAS_ROS) && VESSEL_HAS_ROS
#include <ros/ros.h>
#define VCL_INFO(msg) ROS_INFO_STREAM(msg)
#define VCL_WARN(msg) ROS_WARN_STREAM(msg)
#define VCL_ERROR(msg) ROS_ERROR_STREAM(msg)
#define VCL_WARN_THROTTLE(period, msg) ROS_WARN_THROTTLE(period, ROSCONSOLE_DEFAULT_NAME, msg)
#else
#define VCL_INFO(msg) std::cout << "[INFO] " << msg << std::endl
#define VCL_WARN(msg) std::cerr << "[WARN] " << msg << std::endl
#define VCL_ERROR(msg) std::cerr << "[ERROR] " << msg << std::endl
#define VCL_WARN_THROTTLE(period, msg) VCL_WARN(msg)
#endif
