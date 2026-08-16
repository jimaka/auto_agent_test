#include "vessel_control/koopman_lift.hpp"

#include <ros/ros.h>

namespace vessel_control {

bool KoopmanLift::load(const std::string& model_dir) {
  ROS_INFO_STREAM("KoopmanLift: load model from " << model_dir);
  // TODO: load meta.yaml, norm_x.json, encoder.onnx (ONNX Runtime)
  loaded_ = true;
  return loaded_;
}

bool KoopmanLift::lift(const StateVector& x, LiftVector& z) const {
  if (!loaded_) {
    return false;
  }
  // TODO: normalize x, run ONNX, fill z
  for (std::size_t i = 0; i < z.size(); ++i) {
    z[i] = static_cast<double>(i) * 0.0;
  }
  (void)x;
  return true;
}

}  // namespace vessel_control
