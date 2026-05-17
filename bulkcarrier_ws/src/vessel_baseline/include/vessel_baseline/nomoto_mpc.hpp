#pragma once

#include <array>

namespace vessel_baseline {

struct NomotoParams {
  double K{0.5};
  double T{20.0};
};

struct BaselineSolution {
  std::array<double, 2> u0{{0.0, 0.0}};
  bool success{false};
};

class NomotoMpc {
 public:
  BaselineSolution solve(double e_cross, double e_psi, double e_u,
                         double delta_prev, double rpm_prev) const;
};

}  // namespace vessel_baseline
