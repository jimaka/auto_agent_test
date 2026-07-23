#include "vessel_control/mpc_osqp.hpp"

#include <ros/ros.h>

namespace vessel_control {

bool MpcOsqp::configure(const std::string& model_dir, int horizon_N) {
  ROS_INFO_STREAM("MpcOsqp: configure model=" << model_dir << " N=" << horizon_N);
  // TODO: load A.bin, B.bin, build sparse P, A_con; init OSQP workspace
  return true;
}

MpcSolution MpcOsqp::solve(const StateVector& x, const LiftVector& z,
                             const InputVector& u_prev) {
  MpcSolution sol;
  (void)x;
  (void)z;
  (void)u_prev;
  sol.u0 = {0.0, 0.0};
  sol.success = true;
  sol.status = "stub";
  sol.solve_time_ms = 0.0;
  return sol;
}

}  // namespace vessel_control
