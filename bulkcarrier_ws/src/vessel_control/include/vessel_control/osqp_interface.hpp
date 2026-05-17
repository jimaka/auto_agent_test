#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace vessel_control {

struct OsqpProblem {
  int n{0};
  int m{0};
  std::vector<double> P_upper;  // upper triangular packed or full n*n used upper
  std::vector<double> q;
  std::vector<int> A_row_ptr;
  std::vector<int> A_col_idx;
  std::vector<double> A_data;
  std::vector<double> l;
  std::vector<double> u;
};

struct OsqpResult {
  bool success{false};
  std::string status;
  std::vector<double> x;
  int iterations{0};
};

class OsqpSolver {
 public:
  OsqpSolver();
  ~OsqpSolver();

  OsqpSolver(const OsqpSolver&) = delete;
  OsqpSolver& operator=(const OsqpSolver&) = delete;

  bool setup(const OsqpProblem& problem);
  OsqpResult solve();

 private:
  void cleanup();
  struct Impl;
  Impl* impl_{nullptr};
};

}  // namespace vessel_control
