#include "vessel_control/koopman_lift.hpp"
#include "vessel_control/mpc_osqp.hpp"

#include <iostream>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: mpc_standalone_test <model_dir>\n";
    return 1;
  }
  const std::string model_dir = argv[1];

  vessel_control::KoopmanLift lift;
  if (!lift.load(model_dir)) {
    std::cerr << "Failed to load lift\n";
    return 2;
  }

  vessel_control::MpcOsqp mpc;
  mpc.set_lift(&lift);
  if (!mpc.configure(lift.bundle(), lift.bundle().horizon_N)) {
    std::cerr << "Failed to configure MPC\n";
    return 3;
  }

  vessel_control::StateVector x{0, 0, 0, 5, 0, 0};
  vessel_control::LiftVector z{};
  if (!lift.lift(x, z)) {
    std::cerr << "Lift failed\n";
    return 4;
  }

  vessel_control::InputVector u_prev{0.0, 60.0};
  std::vector<vessel_control::StateVector> refs(30, x);
  refs[10][0] = 10.0;

  const auto sol = mpc.solve(x, z, u_prev, refs);
  std::cout << "success=" << sol.success << " status=" << sol.status << " ms=" << sol.solve_time_ms
            << " delta=" << sol.u0[0] << " rpm=" << sol.u0[1] << "\n";
  return sol.success ? 0 : 5;
}
