#pragma once

#include "vessel_control/types.hpp"
#include <vessel_msgs/VesselState.h>
#include <vessel_msgs/Wind.h>

namespace vessel_control {

/// Build x = [x, y, psi, u, v, r] from INS (body-frame u,v,r).
StateVector assemble_state(const vessel_msgs::VesselState& ins);

}  // namespace vessel_control
