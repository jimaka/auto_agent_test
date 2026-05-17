#pragma once

#include <array>
#include <string>

namespace vessel_control {

constexpr int NX = 6;
constexpr int NU = 2;
constexpr int NZ = 64;
constexpr int DEFAULT_HORIZON = 30;

using StateVector = std::array<double, NX>;
using InputVector = std::array<double, NU>;
using LiftVector = std::array<double, NZ>;

struct ModelMeta {
  std::string model_id;
  double Ts{0.25};
  int nz{NZ};
  int horizon_N{DEFAULT_HORIZON};
};

}  // namespace vessel_control
