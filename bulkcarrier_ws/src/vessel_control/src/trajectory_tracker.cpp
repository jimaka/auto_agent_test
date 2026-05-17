#include "vessel_control/trajectory_tracker.hpp"

#include <cmath>

namespace vessel_control {

static double wrap_pi(double a) {
  while (a > M_PI) a -= 2.0 * M_PI;
  while (a < -M_PI) a += 2.0 * M_PI;
  return a;
}

bool lookup_reference(const vessel_msgs::TrajectoryRef& traj, const ros::Time& t,
                      StateVector& x_ref, TrackingErrors* errors) {
  if (traj.points.empty()) {
    return false;
  }
  const double tk = t.toSec();
  const auto& p = traj.points.back();
  for (const auto& pt : traj.points) {
    if (pt.t >= tk) {
      x_ref = {pt.x, pt.y, pt.psi, pt.u, pt.v, pt.r};
      if (errors) {
        errors->e_cross = 0.0;
        errors->e_psi = 0.0;
        errors->e_u = 0.0;
      }
      return true;
    }
  }
  x_ref = {p.x, p.y, p.psi, p.u, p.v, p.r};
  (void)errors;
  (void)wrap_pi;
  return true;
}

}  // namespace vessel_control
