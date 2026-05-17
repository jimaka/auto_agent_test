#include "vessel_control/mpc_osqp.hpp"
#include "vessel_control/koopman_lift.hpp"

#include "vessel_control/log.hpp"

#include <chrono>
#include <cmath>
#include <cstring>

namespace vessel_control {
namespace {

constexpr double kPi = 3.14159265358979323846;

void mat_mul(const std::vector<double>& A, int rows, int cols, const std::vector<double>& B, int bcols,
             std::vector<double>& C) {
  C.assign(static_cast<std::size_t>(rows * bcols), 0.0);
  for (int i = 0; i < rows; ++i) {
    for (int k = 0; k < cols; ++k) {
      const double a = A[static_cast<std::size_t>(i * cols + k)];
      for (int j = 0; j < bcols; ++j) {
        C[static_cast<std::size_t>(i * bcols + j)] += a * B[static_cast<std::size_t>(k * bcols + j)];
      }
    }
  }
}

void mat_vec(const std::vector<double>& A, int rows, int cols, const std::vector<double>& x,
             std::vector<double>& y) {
  y.assign(static_cast<std::size_t>(rows), 0.0);
  for (int i = 0; i < rows; ++i) {
    double s = 0.0;
    for (int j = 0; j < cols; ++j) {
      s += A[static_cast<std::size_t>(i * cols + j)] * x[static_cast<std::size_t>(j)];
    }
    y[static_cast<std::size_t>(i)] = s;
  }
}

void mat_transpose_mul(const std::vector<double>& A, int rows, int cols, const std::vector<double>& B,
                       int brows, int bcols, std::vector<double>& C) {
  // C = A^T B, A is rows x cols
  C.assign(static_cast<std::size_t>(cols * bcols), 0.0);
  for (int i = 0; i < cols; ++i) {
    for (int j = 0; j < bcols; ++j) {
      double s = 0.0;
      for (int k = 0; k < rows; ++k) {
        s += A[static_cast<std::size_t>(k * cols + i)] * B[static_cast<std::size_t>(k * bcols + j)];
      }
      C[static_cast<std::size_t>(i * bcols + j)] = s;
    }
  }
}

std::vector<double> mat_transpose(const std::vector<double>& A, int rows, int cols) {
  std::vector<double> AT(static_cast<std::size_t>(rows * cols));
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      AT[static_cast<std::size_t>(j * rows + i)] = A[static_cast<std::size_t>(i * cols + j)];
    }
  }
  return AT;
}

void add_regularized(std::vector<double>& M, int n, double eps) {
  for (int i = 0; i < n; ++i) {
    M[static_cast<std::size_t>(i * n + i)] += eps;
  }
}

}  // namespace

MpcOsqp::MpcOsqp() = default;
MpcOsqp::~MpcOsqp() = default;

bool MpcOsqp::configure(const std::string& model_dir, int horizon_N) {
  ModelBundle bundle;
  if (!load_model_bundle(model_dir, bundle)) return false;
  return configure(bundle, horizon_N);
}

bool MpcOsqp::configure(const ModelBundle& bundle, int horizon_N) {
  bundle_ = bundle;
  horizon_ = horizon_N > 0 ? horizon_N : bundle_.horizon_N;
  nz_ = bundle_.nz;
  n_vars_ = horizon_ * NU;
  configured_ = false;
  solver_ready_ = false;

  if (tube_enabled_) {
    const std::string tube_path = bundle_.model_dir + "/tube_tightening.csv";
    TubeTightening tube;
    if (load_tube_csv(tube_path, tube)) {
      tube_ = tube;
    }
  }

  Qz_.assign(static_cast<std::size_t>(nz_ * nz_), 0.0);
  if (bundle_.mats.has_cx) {
    const auto& Cx = bundle_.mats.Cx;
    const double qx[NX] = {bundle_.weights.w_cross, bundle_.weights.w_cross, bundle_.weights.w_psi,
                           bundle_.weights.w_u, 0.1, 0.1};
    for (int i = 0; i < nz_; ++i) {
      for (int j = 0; j < nz_; ++j) {
        double s = 0.0;
        for (int r = 0; r < NX; ++r) {
          s += Cx[static_cast<std::size_t>(r * nz_ + i)] * qx[r] * Cx[static_cast<std::size_t>(r * nz_ + j)];
        }
        Qz_[static_cast<std::size_t>(i * nz_ + j)] = s;
      }
    }
  } else {
    for (int i = 0; i < nz_; ++i) {
      Qz_[static_cast<std::size_t>(i * nz_ + i)] = 1.0;
    }
  }

  R_.assign(static_cast<std::size_t>(NU * NU), 0.0);
  R_[0] = bundle_.weights.w_delta;
  R_[static_cast<std::size_t>(NU + 1)] = bundle_.weights.w_n;

  if (!build_prediction_mats()) return false;
  configured_ = true;
  VCL_INFO("MpcOsqp: configured N=" << horizon_ << " nz=" << nz_ << " n_u=" << n_vars_);
  return true;
}

bool MpcOsqp::build_prediction_mats() {
  const int N = horizon_;
  const int nz = nz_;
  const int nu = NU;

  std::vector<double> Apow(static_cast<std::size_t>((N + 1) * nz * nz), 0.0);
  for (int i = 0; i < nz; ++i) {
    Apow[static_cast<std::size_t>(i * nz + i)] = 1.0;  // A^0
  }
  std::vector<double> A = bundle_.mats.A;
  std::vector<double> Ak = Apow;
  for (int p = 1; p <= N; ++p) {
    std::vector<double> Ak_next;
    mat_mul(A, nz, nz, Ak, nz, Ak_next);
    Ak = Ak_next;
    std::copy(Ak.begin(), Ak.end(), &Apow[static_cast<std::size_t>(p * nz * nz)]);
  }

  Phi_.assign(static_cast<std::size_t>(N * nz * nz), 0.0);
  for (int k = 0; k < N; ++k) {
    std::memcpy(&Phi_[static_cast<std::size_t>(k * nz * nz)], &Apow[static_cast<std::size_t>((k + 1) * nz * nz)],
                static_cast<std::size_t>(nz * nz) * sizeof(double));
  }

  Psi_.assign(static_cast<std::size_t>(N * nz * N * nu), 0.0);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j <= i; ++j) {
      const int power = i - j;
      std::vector<double> term;
      mat_mul(&Apow[static_cast<std::size_t>(power * nz * nz)], nz, nz, bundle_.mats.B, nu, term);
      for (int r = 0; r < nz; ++r) {
        for (int c = 0; c < nu; ++c) {
          Psi_[static_cast<std::size_t>(i * nz * (N * nu) + r * (N * nu) + j * nu + c)] =
              term[static_cast<std::size_t>(r * nu + c)];
        }
      }
    }
  }
  return true;
}

bool MpcOsqp::build_osqp_problem(const std::vector<double>& z0,
                                 const std::vector<StateVector>& x_refs, const InputVector& u_prev,
                                 OsqpProblem& prob) const {
  if (!configured_) return false;
  const int N = horizon_;
  const int nz = nz_;
  const int nu = NU;
  const int n = n_vars_;

  std::vector<double> z_ref_stack(static_cast<std::size_t>(N * nz), 0.0);
  for (int k = 0; k < N; ++k) {
    StateVector xref = x_refs[static_cast<std::size_t>(std::min(k, static_cast<int>(x_refs.size()) - 1))];
    std::vector<double> zk(nz);
    if (lift_ && lift_->lift(xref, zk)) {
      std::copy(zk.begin(), zk.end(), &z_ref_stack[static_cast<std::size_t>(k * nz)]);
    }
  }

  std::vector<double> phi_z0;
  mat_vec(Phi_, N * nz, nz, z0, phi_z0);

  std::vector<double> Qbar(static_cast<std::size_t>(N * nz * N * nz), 0.0);
  for (int b = 0; b < N; ++b) {
    for (int i = 0; i < nz; ++i) {
      for (int j = 0; j < nz; ++j) {
        Qbar[static_cast<std::size_t>((b * nz + i) * (N * nz) + (b * nz + j))] =
            Qz_[static_cast<std::size_t>(i * nz + j)];
      }
    }
  }

  std::vector<double> PsiTPsi;
  mat_transpose_mul(Psi_, N * nz, N * nu, Psi_, N * nz, N * nu, PsiTPsi);

  std::vector<double> Rbar(static_cast<std::size_t>(N * nu * N * nu), 0.0);
  for (int b = 0; b < N; ++b) {
    for (int i = 0; i < nu; ++i) {
      for (int j = 0; j < nu; ++j) {
        Rbar[static_cast<std::size_t>((b * nu + i) * (N * nu) + (b * nu + j))] = R_[static_cast<std::size_t>(i * nu + j)];
      }
    }
  }

  prob.P_upper.assign(static_cast<std::size_t>(n * n), 0.0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      prob.P_upper[static_cast<std::size_t>(i * n + j)] =
          2.0 * (PsiTPsi[static_cast<std::size_t>(i * n + j)] + Rbar[static_cast<std::size_t>(i * n + j)]);
    }
  }
  add_regularized(prob.P_upper, n, 1e-6);

  std::vector<double> err = phi_z0;
  for (int i = 0; i < N * nz; ++i) {
    err[static_cast<std::size_t>(i)] -= z_ref_stack[static_cast<std::size_t>(i)];
  }
  std::vector<double> PsiT_err;
  mat_transpose_mul(Psi_, N * nz, N * nu, err, N * nz, 1, PsiT_err);
  prob.q.resize(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    prob.q[static_cast<std::size_t>(i)] = 2.0 * PsiT_err[static_cast<std::size_t>(i)];
  }

  auto norm_u = [&](double u_phys, int j) {
    return (u_phys - bundle_.norm.mu_u[j]) / bundle_.norm.sigma_u[j];
  };

  const int m = N * nu;
  prob.m = m;
  prob.n = n;
  prob.A_row_ptr.assign(static_cast<std::size_t>(m + 1), 0);
  prob.A_col_idx.clear();
  prob.A_data.clear();
  prob.l.assign(static_cast<std::size_t>(m), 0.0);
  prob.u.assign(static_cast<std::size_t>(m), 0.0);

  int nnz = 0;
  int row = 0;
  for (int k = 0; k < N; ++k) {
    for (int j = 0; j < nu; ++j) {
      const int col = k * nu + j;
      const double umin = (j == 0) ? bundle_.constraints.delta_min : bundle_.constraints.n_min;
      const double umax = (j == 0) ? bundle_.constraints.delta_max : bundle_.constraints.n_max;
      prob.A_row_ptr[static_cast<std::size_t>(row)] = nnz;
      prob.A_col_idx.push_back(col);
      prob.A_data.push_back(1.0);
      prob.l[static_cast<std::size_t>(row)] = norm_u(umin, j);
      prob.u[static_cast<std::size_t>(row)] = norm_u(umax, j);
      ++nnz;
      ++row;
    }
  }
  (void)u_prev;
  prob.A_row_ptr[static_cast<std::size_t>(m)] = nnz;
  return true;
}

MpcSolution MpcOsqp::solve(const StateVector& x, const LiftVector& z, const InputVector& u_prev,
                           const std::vector<StateVector>& x_refs) const {
  MpcSolution sol;
  if (!configured_) {
    sol.status = "not_configured";
    return sol;
  }

  std::vector<double> z0(static_cast<std::size_t>(nz_));
  if (static_cast<int>(z.size()) >= nz_) {
    for (int i = 0; i < nz_; ++i) z0[static_cast<std::size_t>(i)] = z[static_cast<std::size_t>(i)];
  } else if (lift_ && !lift_->lift(x, z0)) {
    sol.status = "lift_failed";
    return sol;
  }

  std::vector<StateVector> refs = x_refs;
  if (refs.empty()) {
    refs.assign(static_cast<std::size_t>(horizon_), x);
  }

  const auto t0 = std::chrono::steady_clock::now();
  OsqpProblem problem;
  if (!build_osqp_problem(z0, refs, u_prev, problem)) {
    sol.status = "build_failed";
    return sol;
  }

  OsqpSolver& solver = const_cast<OsqpSolver&>(solver_);
  if (!solver.setup(problem)) {
    sol.status = "osqp_setup_failed";
    return sol;
  }

  const OsqpResult res = solver.solve();
  const auto t1 = std::chrono::steady_clock::now();
  sol.solve_time_ms =
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
  sol.success = res.success;
  sol.status = res.status;
  if (res.success && static_cast<int>(res.x.size()) >= NU) {
    for (int j = 0; j < NU; ++j) {
      sol.u0[j] = res.x[static_cast<std::size_t>(j)] * bundle_.norm.sigma_u[j] + bundle_.norm.mu_u[j];
    }
  }
  return sol;
}

}  // namespace vessel_control
