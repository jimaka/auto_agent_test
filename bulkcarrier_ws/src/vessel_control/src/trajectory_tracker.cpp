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
  if (tk <= traj.points.front().t) {
    const auto& p0 = traj.points.front();
    x_ref = {p0.x, p0.y, p0.psi, p0.u, p0.v, p0.r};
    return true;
  }
  if (tk >= traj.points.back().t) {
    const auto& pN = traj.points.back();
    x_ref = {pN.x, pN.y, pN.psi, pN.u, pN.v, pN.r};
    return true;
  }
  for (std::size_t i = 0; i + 1 < traj.points.size(); ++i) {
    const auto& a = traj.points[i];
    const auto& b = traj.points[i + 1];
    if (tk >= a.t && tk <= b.t) {
      const double alpha = (tk - a.t) / std::max(b.t - a.t, 1e-9);
      x_ref = {a.x + alpha * (b.x - a.x),     a.y + alpha * (b.y - a.y),
               a.psi + alpha * (b.psi - a.psi), a.u + alpha * (b.u - a.u),
               a.v + alpha * (b.v - a.v),     a.r + alpha * (b.r - a.r)};
      x_ref[2] = wrap_pi(x_ref[2]);
      (void)errors;
      return true;
    }
  }
  return false;
}

}  // namespace vessel_control
