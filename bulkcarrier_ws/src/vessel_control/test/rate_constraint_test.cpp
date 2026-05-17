#include "vessel_control/koopman_lift.hpp"
#include "vessel_control/mpc_osqp.hpp"

#include <cmath>
#include <iostream>

int main(int argc, char** argv) {
  const std::string model_dir =
      argc > 1 ? argv[1] : "../model_registry/koopman_test";

  vessel_control::KoopmanLift lift;
  if (!lift.load(model_dir)) {
    std::cerr << "load failed\n";
    return 1;
  }

  vessel_control::ModelBundle bundle = lift.bundle();
  bundle.constraints.delta_rate_max = 0.035;
  bundle.constraints.n_rate_max = 5.0;
  bundle.Ts = 0.25;

  vessel_control::MpcOsqp mpc;
  mpc.set_lift(&lift);
  if (!mpc.configure(bundle, bundle.horizon_N)) {
    std::cerr << "configure failed\n";
    return 2;
  }

  vessel_control::StateVector x{0, 0, 0, 5, 0, 0};
  vessel_control::LiftVector z{};
  lift.lift(x, z);

  vessel_control::InputVector u_prev{0.0, 60.0};
  std::vector<vessel_control::StateVector> refs(bundle.horizon_N, x);
  refs.back()[0] = 50.0;

  const auto sol = mpc.solve(x, z, u_prev, refs);
  if (!sol.success) {
    std::cerr << "solve failed: " << sol.status << "\n";
    return 3;
  }

  const double d_delta = std::abs(sol.u0[0] - u_prev[0]);
  const double d_rpm = std::abs(sol.u0[1] - u_prev[1]);
  const double lim_delta = bundle.constraints.delta_rate_max * bundle.Ts + 1e-6;
  const double lim_rpm = bundle.constraints.n_rate_max * bundle.Ts + 1e-6;

  std::cout << "d_delta=" << d_delta << " lim=" << lim_delta << "\n";
  std::cout << "d_rpm=" << d_rpm << " lim=" << lim_rpm << "\n";

  if (d_delta > lim_delta + 1e-4) {
    std::cerr << "rudder rate constraint violated\n";
    return 4;
  }
  if (d_rpm > lim_rpm + 1e-4) {
    std::cerr << "rpm rate constraint violated\n";
    return 5;
  }
  std::cout << "rate constraints OK\n";
  return 0;
}
