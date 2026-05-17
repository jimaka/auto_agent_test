#include "vessel_baseline/nomoto_mpc.hpp"

namespace vessel_baseline {

BaselineSolution NomotoMpc::solve(double e_cross, double e_psi, double e_u,
                                   double delta_prev, double rpm_prev) const {
  (void)delta_prev;
  BaselineSolution sol;
  // TODO: linear MPC / LQR on Nomoto surge-yaw surrogate
  sol.u0[0] = -0.1 * e_psi - 0.05 * e_cross;
  sol.u0[1] = rpm_prev + 2.0 * e_u;
  sol.success = true;
  return sol;
}

}  // namespace vessel_baseline
