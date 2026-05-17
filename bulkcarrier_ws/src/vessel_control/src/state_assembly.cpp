#include "vessel_control/state_assembly.hpp"

namespace vessel_control {

StateVector assemble_state(const vessel_msgs::VesselState& ins) {
  return {ins.x, ins.y, ins.psi, ins.u, ins.v, ins.r};
}

}  // namespace vessel_control
