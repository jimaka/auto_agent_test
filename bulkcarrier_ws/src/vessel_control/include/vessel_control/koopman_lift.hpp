#pragma once

#include "vessel_control/model_loader.hpp"
#include "vessel_control/types.hpp"

#include <memory>
#include <string>
#include <vector>

namespace vessel_control {

class KoopmanLift {
 public:
  KoopmanLift();
  ~KoopmanLift();

  bool load(const std::string& model_dir);
  bool lift(const StateVector& x, std::vector<double>& z) const;
  bool lift(const StateVector& x, LiftVector& z) const;

  int nz() const { return nz_; }
  const ModelBundle& bundle() const { return bundle_; }

 private:
  std::array<double, NX> normalize_x(const StateVector& x) const;

  ModelBundle bundle_{};
  int nz_{NZ};
  bool loaded_{false};

  struct OnnxImpl;
  std::unique_ptr<OnnxImpl> onnx_;
};

}  // namespace vessel_control
