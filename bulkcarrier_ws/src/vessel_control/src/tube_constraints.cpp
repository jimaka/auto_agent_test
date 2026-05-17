#include "vessel_control/tube_constraints.hpp"

#include <fstream>
#include <sstream>

namespace vessel_control {

bool load_tube_csv(const std::string& path, TubeTightening& tube) {
  std::ifstream in(path);
  if (!in.is_open()) {
    return false;
  }
  std::string line;
  std::getline(in, line);  // header
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::stringstream ss(line);
    char comma;
    int step;
    double ec, ep, eu;
    if (ss >> step >> comma >> ec >> comma >> ep >> comma >> eu) {
      tube.eps_cross.push_back(ec);
      tube.eps_psi.push_back(ep);
      tube.eps_u.push_back(eu);
    }
  }
  return !tube.eps_cross.empty();
}

}  // namespace vessel_control
