#pragma once

#include "vessel_control/types.hpp"
#include <vessel_msgs/TrajectoryRef.h>
#include <ros/time.h>

namespace vessel_control {

struct TrackingErrors {
  double e_cross{0.0};
  double e_psi{0.0};
  double e_u{0.0};
};

bool lookup_reference(const vessel_msgs::TrajectoryRef& traj, const ros::Time& t,
                      StateVector& x_ref, TrackingErrors* errors = nullptr);

}  // namespace vessel_control
