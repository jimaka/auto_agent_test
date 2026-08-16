#pragma once

#include "vessel_control/types.hpp"
#include <string>

namespace vessel_control {

struct MpcSolution {
  InputVector u0{};
  bool success{false};
  double solve_time_ms{0.0};
  std::string status{"not_solved"};
};

class MpcOsqp {
 public:
  bool configure(const std::string& model_dir, int horizon_N);
  MpcSolution solve(const StateVector& x, const LiftVector& z,
                    const InputVector& u_prev);
};

}  // namespace vessel_control
