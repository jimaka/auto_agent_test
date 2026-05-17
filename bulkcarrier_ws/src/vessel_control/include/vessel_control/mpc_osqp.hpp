#pragma once

#include "vessel_control/model_loader.hpp"

namespace vessel_control {
class KoopmanLift;
}
#include "vessel_control/osqp_interface.hpp"
#include "vessel_control/tube_constraints.hpp"
#include "vessel_control/types.hpp"

#include <memory>
#include <string>
#include <vector>

namespace vessel_control {

struct MpcSolution {
  InputVector u0{};
  bool success{false};
  double solve_time_ms{0.0};
  std::string status{"not_solved"};
};

class MpcOsqp {
 public:
  MpcOsqp();
  ~MpcOsqp();

  bool configure(const std::string& model_dir, int horizon_N);
  bool configure(const ModelBundle& bundle, int horizon_N);

  void set_lift(KoopmanLift* lift) { lift_ = lift; }

  MpcSolution solve(const StateVector& x, const LiftVector& z, const InputVector& u_prev,
                    const std::vector<StateVector>& x_refs) const;

 private:
  bool build_prediction_mats();
  bool build_osqp_problem(const std::vector<double>& z0,
                          const std::vector<StateVector>& x_refs, const InputVector& u_prev,
                          OsqpProblem& prob) const;

  ModelBundle bundle_{};
  int horizon_{DEFAULT_HORIZON};
  int nz_{NZ};
  int n_vars_{0};
  bool configured_{false};
  bool tube_enabled_{true};
  TubeTightening tube_{};
  KoopmanLift* lift_{nullptr};

  std::vector<double> Psi_;   // (N*nz) x (N*nu) row-major
  std::vector<double> Phi_;   // (N*nz) x nz
  std::vector<double> Qz_;    // nz x nz
  std::vector<double> R_;     // nu x nu

  mutable OsqpSolver solver_;
  mutable bool solver_ready_{false};
  mutable OsqpProblem last_problem_;
};

}  // namespace vessel_control
