#include "vessel_control/koopman_lift.hpp"

#include "vessel_control/log.hpp"

#ifdef VESSEL_USE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

#include <array>
#include <cmath>

namespace vessel_control {

#ifdef VESSEL_USE_ONNXRUNTIME
struct KoopmanLift::OnnxImpl {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "vessel_control"};
  Ort::SessionOptions options;
  std::unique_ptr<Ort::Session> session;
  std::vector<const char*> input_names{"x_norm"};
  std::vector<const char*> output_names{"z"};
};
#else
struct KoopmanLift::OnnxImpl {};
#endif

KoopmanLift::KoopmanLift() : onnx_(std::make_unique<OnnxImpl>()) {}

KoopmanLift::~KoopmanLift() = default;

std::array<double, NX> KoopmanLift::normalize_x(const StateVector& x) const {
  std::array<double, NX> out{};
  for (int i = 0; i < NX; ++i) {
    out[i] = (x[i] - bundle_.norm.mu[i]) / bundle_.norm.sigma[i];
  }
  return out;
}

bool KoopmanLift::load(const std::string& model_dir) {
  loaded_ = false;
  if (!load_model_bundle(model_dir, bundle_)) {
    VCL_ERROR("KoopmanLift: failed to load model bundle from " << model_dir);
    return false;
  }
  nz_ = bundle_.nz;

#ifdef VESSEL_USE_ONNXRUNTIME
  try {
    onnx_->options.SetIntraOpNumThreads(1);
    onnx_->options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
#ifdef _WIN32
    const std::wstring path(bundle_.encoder_onnx_path.begin(), bundle_.encoder_onnx_path.end());
    onnx_->session = std::make_unique<Ort::Session>(onnx_->env, path.c_str(), onnx_->options);
#else
    onnx_->session =
        std::make_unique<Ort::Session>(onnx_->env, bundle_.encoder_onnx_path.c_str(), onnx_->options);
#endif
    loaded_ = true;
    VCL_INFO("KoopmanLift: ONNX encoder loaded, nz=" << nz_);
    return true;
  } catch (const std::exception& e) {
    VCL_ERROR("KoopmanLift: ONNX load failed: " << e.what());
    return false;
  }
#else
  VCL_ERROR("KoopmanLift: built without ONNX Runtime (VESSEL_USE_ONNXRUNTIME=OFF)");
  return false;
#endif
}

bool KoopmanLift::lift(const StateVector& x, std::vector<double>& z) const {
  if (!loaded_) return false;
  z.assign(static_cast<std::size_t>(nz_), 0.0);
  const auto xn = normalize_x(x);

#ifdef VESSEL_USE_ONNXRUNTIME
  try {
    std::array<int64_t, 2> shape{1, NX};
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input = Ort::Value::CreateTensor<double>(mem, const_cast<double*>(xn.data()), NX,
                                                        shape.data(), shape.size());
    auto outputs = onnx_->session->Run(Ort::RunOptions{nullptr}, onnx_->input_names.data(), &input, 1,
                                       onnx_->output_names.data(), 1);
    const double* zraw = outputs[0].GetTensorData<double>();
    auto zshape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const int out_dim = zshape.size() > 1 ? static_cast<int>(zshape[1]) : static_cast<int>(zshape[0]);
    const int copy_n = std::min(nz_, out_dim);
    for (int i = 0; i < copy_n; ++i) z[static_cast<std::size_t>(i)] = zraw[i];
    return true;
  } catch (const std::exception& e) {
    VCL_ERROR("KoopmanLift: inference failed: " << e.what());
    return false;
  }
#else
  (void)xn;
  return false;
#endif
}

bool KoopmanLift::lift(const StateVector& x, LiftVector& z) const {
  std::vector<double> zv;
  if (!lift(x, zv)) return false;
  z.fill(0.0);
  const int n = std::min(nz_, NZ);
  for (int i = 0; i < n; ++i) z[static_cast<std::size_t>(i)] = zv[static_cast<std::size_t>(i)];
  return true;
}

}  // namespace vessel_control
