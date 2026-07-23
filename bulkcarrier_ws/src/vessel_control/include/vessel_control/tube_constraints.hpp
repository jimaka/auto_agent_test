#pragma once

#include <string>
#include <vector>

namespace vessel_control {

struct TubeTightening {
  std::vector<double> eps_cross;
  std::vector<double> eps_psi;
  std::vector<double> eps_u;
};

bool load_tube_csv(const std::string& path, TubeTightening& tube);

}  // namespace vessel_control
