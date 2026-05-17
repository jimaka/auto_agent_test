#pragma once

#include "vessel_control/types.hpp"
#include <string>

namespace vessel_control {

class KoopmanLift {
 public:
  bool load(const std::string& model_dir);
  bool lift(const StateVector& x, LiftVector& z) const;

 private:
  bool loaded_{false};
};

}  // namespace vessel_control
