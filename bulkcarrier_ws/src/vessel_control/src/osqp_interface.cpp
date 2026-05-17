#include "vessel_control/osqp_interface.hpp"

#include "vessel_control/log.hpp"

#ifdef VESSEL_USE_OSQP
#include <osqp.h>
#endif

#include <cstring>
#include <stdexcept>

namespace vessel_control {

struct OsqpSolver::Impl {
#ifdef VESSEL_USE_OSQP
  OSQPWorkspace* work{nullptr};
  OSQPSettings settings;
  OSQPData data;
  std::vector<double> P_x;
  std::vector<c_int> P_i;
  std::vector<c_int> P_p;
  std::vector<double> A_x;
  std::vector<c_int> A_i;
  std::vector<c_int> A_p;
  std::vector<double> q;
  std::vector<double> l;
  std::vector<double> u;
#endif
};

OsqpSolver::OsqpSolver() : impl_(new Impl()) {}

OsqpSolver::~OsqpSolver() {
  cleanup();
  delete impl_;
}

void OsqpSolver::cleanup() {
#ifdef VESSEL_USE_OSQP
  if (impl_->work) {
    osqp_cleanup(impl_->work);
    impl_->work = nullptr;
  }
  if (impl_->data.A) {
    c_free(impl_->data.A);
    impl_->data.A = nullptr;
  }
  if (impl_->data.P) {
    c_free(impl_->data.P);
    impl_->data.P = nullptr;
  }
#endif
}

#ifdef VESSEL_USE_OSQP
static void dense_to_csc_upper(const std::vector<double>& dense, int n,
                               std::vector<double>& x, std::vector<c_int>& i, std::vector<c_int>& p) {
  x.clear();
  i.clear();
  p.assign(static_cast<std::size_t>(n) + 1, 0);
  int nnz = 0;
  for (int col = 0; col < n; ++col) {
    p[col] = nnz;
    for (int row = 0; row <= col; ++row) {
      const double v = dense[static_cast<std::size_t>(row) * n + col];
      if (row == col || std::abs(v) > 1e-12) {
        x.push_back(v);
        i.push_back(row);
        ++nnz;
      }
    }
  }
  p[n] = nnz;
}
#endif

bool OsqpSolver::setup(const OsqpProblem& problem) {
  cleanup();
#ifdef VESSEL_USE_OSQP
  const c_int n = problem.n;
  const c_int m = problem.m;

  dense_to_csc_upper(problem.P_upper, n, impl_->P_x, impl_->P_i, impl_->P_p);
  impl_->P_p = impl_->P_p;  // keep

  impl_->A_x = problem.A_data;
  impl_->A_i.assign(problem.A_col_idx.begin(), problem.A_col_idx.end());
  impl_->A_p.assign(problem.A_row_ptr.begin(), problem.A_row_ptr.end());
  impl_->q = problem.q;
  impl_->l = problem.l;
  impl_->u = problem.u;

  impl_->data.n = n;
  impl_->data.m = m;
  impl_->data.P = csc_matrix(n, n, static_cast<c_int>(impl_->P_x.size()), impl_->P_x.data(), impl_->P_i.data(),
                              impl_->P_p.data());
  impl_->data.A = csc_matrix(m, n, static_cast<c_int>(impl_->A_x.size()), impl_->A_x.data(), impl_->A_i.data(),
                              impl_->A_p.data());
  impl_->data.q = impl_->q.data();
  impl_->data.l = impl_->l.data();
  impl_->data.u = impl_->u.data();

  osqp_set_default_settings(&impl_->settings);
  impl_->settings.verbose = 0;
  impl_->settings.warm_start = 1;
  impl_->settings.max_iter = 4000;
  impl_->settings.eps_abs = 1e-4;
  impl_->settings.eps_rel = 1e-4;

  const c_int setup_status = osqp_setup(&impl_->work, &impl_->data, &impl_->settings);
  return setup_status == 0;
#else
  (void)problem;
  VCL_WARN_THROTTLE(5.0, "vessel_control built without OSQP (VESSEL_USE_OSQP=OFF)");
  return false;
#endif
}

OsqpResult OsqpSolver::solve() {
  OsqpResult out;
#ifdef VESSEL_USE_OSQP
  if (!impl_->work) {
    out.status = "not_setup";
    return out;
  }
  const c_int status = osqp_solve(impl_->work);
  if (status != 0) {
    out.status = "solve_error";
    return out;
  }
  out.success = impl_->work->info->status_val == OSQP_SOLVED || impl_->work->info->status_val == OSQP_SOLVED_INACCURATE;
  out.status = impl_->work->info->status;
  out.iterations = static_cast<int>(impl_->work->info->iter);
  out.x.assign(impl_->work->solution->x, impl_->work->solution->x + impl_->data.n);
#else
  out.status = "osqp_disabled";
#endif
  return out;
}

}  // namespace vessel_control
