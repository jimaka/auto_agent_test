#pragma once

#include "vessel_control/types.hpp"

#include <array>
#include <string>
#include <vector>

namespace vessel_control {

struct NormParams {
  std::array<double, NX> mu{};
  std::array<double, NX> sigma{};
  std::array<double, NU> mu_u{};
  std::array<double, NU> sigma_u{};
};

struct KoopmanMatrices {
  int nz{NZ};
  int nu{NU};
  std::vector<double> A;  // row-major nz*nz
  std::vector<double> B;  // row-major nz*nu
  std::vector<double> Cx; // row-major nx*nz (optional)
  std::array<double, NX> decoder_bias{};
  bool has_cx{false};
};

struct MpcWeights {
  double w_cross{10.0};
  double w_psi{5.0};
  double w_u{1.0};
  double w_delta{0.1};
  double w_n{0.01};
};

struct InputConstraints {
  double delta_min{-0.61};
  double delta_max{0.61};
  double delta_rate_max{0.035};
  double n_min{0.0};
  double n_max{120.0};
  double n_rate_max{5.0};
};

struct EncoderIoSpec {
  std::string input_name{"x_norm"};
  std::string output_name{"z"};
  int nx{NX};
  int nz{NZ};
  int opset{18};
};

struct ModelBundle {
  std::string model_dir;
  std::string model_id;
  double Ts{0.25};
  int nz{NZ};
  int nx{NX};
  int nu{NU};
  int horizon_N{DEFAULT_HORIZON};
  NormParams norm;
  KoopmanMatrices mats;
  MpcWeights weights;
  InputConstraints constraints;
  std::string encoder_onnx_path;
  EncoderIoSpec encoder_io;
};

bool load_encoder_io_json(const std::string& path, EncoderIoSpec& spec);

bool load_model_bundle(const std::string& model_dir, ModelBundle& out);

std::vector<double> read_f64_bin(const std::string& path, std::size_t expected);

}  // namespace vessel_control
