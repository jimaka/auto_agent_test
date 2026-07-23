#include "vessel_control/model_loader.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace vessel_control {
namespace {

bool parse_yaml_scalar(const std::string& line, const std::string& key, double& out) {
  std::regex re("^\\s*" + key + "\\s*:\\s*([0-9eE+\\.-]+)\\s*$");
  std::smatch m;
  if (std::regex_match(line, m, re)) {
    out = std::stod(m[1].str());
    return true;
  }
  return false;
}

bool parse_yaml_scalar_int(const std::string& line, const std::string& key, int& out) {
  double v;
  if (!parse_yaml_scalar(line, key, v)) return false;
  out = static_cast<int>(v);
  return true;
}

bool parse_json_array(const std::string& text, const std::string& key, std::vector<double>& out) {
  const std::string pat = "\"" + key + "\"";
  auto pos = text.find(pat);
  if (pos == std::string::npos) return false;
  pos = text.find('[', pos);
  if (pos == std::string::npos) return false;
  auto end = text.find(']', pos);
  if (end == std::string::npos) return false;
  std::string slice = text.substr(pos + 1, end - pos - 1);
  std::stringstream ss(slice);
  out.clear();
  std::string tok;
  while (std::getline(ss, tok, ',')) {
      auto b = tok.find_first_not_of(" \t\n\r");
      auto e = tok.find_last_not_of(" \t\n\r");
      if (b == std::string::npos) continue;
      out.push_back(std::stod(tok.substr(b, e - b + 1)));
  }
  return !out.empty();
}

}  // namespace

std::vector<double> read_f64_bin(const std::string& path, std::size_t expected) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("Cannot open bin file: " + path);
  }
  in.seekg(0, std::ios::end);
  const auto nbytes = static_cast<std::size_t>(in.tellg());
  in.seekg(0, std::ios::beg);
  if (nbytes % sizeof(double) != 0) {
    throw std::runtime_error("Invalid bin size: " + path);
  }
  const std::size_t n = nbytes / sizeof(double);
  if (expected > 0 && n != expected) {
    throw std::runtime_error("Unexpected bin length for " + path);
  }
  std::vector<double> data(n);
  in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(nbytes));
  return data;
}

bool load_model_bundle(const std::string& model_dir, ModelBundle& out) {
  out = ModelBundle{};
  out.model_dir = model_dir;

  const std::string meta_path = model_dir + "/meta.yaml";
  std::ifstream meta(meta_path);
  if (!meta) {
    return false;
  }

  std::string line;
  while (std::getline(meta, line)) {
    parse_yaml_scalar_int(line, "nz", out.nz);
    parse_yaml_scalar_int(line, "horizon_N", out.horizon_N);
    parse_yaml_scalar(line, "Ts", out.Ts);
    if (line.find("model_id:") != std::string::npos) {
      auto q1 = line.find('"');
      auto q2 = line.rfind('"');
      if (q1 != std::string::npos && q2 > q1) {
        out.model_id = line.substr(q1 + 1, q2 - q1 - 1);
      } else {
        auto c = line.find(':');
        if (c != std::string::npos) {
          out.model_id = line.substr(c + 1);
          auto b = out.model_id.find_first_not_of(" \t");
          out.model_id.erase(0, b);
        }
      }
    }
    parse_yaml_scalar(line, "w_cross", out.weights.w_cross);
    parse_yaml_scalar(line, "w_psi", out.weights.w_psi);
    parse_yaml_scalar(line, "w_u", out.weights.w_u);
    parse_yaml_scalar(line, "w_delta", out.weights.w_delta);
    parse_yaml_scalar(line, "w_n", out.weights.w_n);
    if (line.find("delta:") != std::string::npos && line.find("min:") != std::string::npos) {
      parse_yaml_scalar(line, "min", out.constraints.delta_min);
    }
    if (line.find("delta:") != std::string::npos && line.find("max:") != std::string::npos) {
      parse_yaml_scalar(line, "max", out.constraints.delta_max);
    }
    if (line.find("n:") != std::string::npos && line.find("min:") != std::string::npos) {
      parse_yaml_scalar(line, "min", out.constraints.n_min);
    }
    if (line.find("n:") != std::string::npos && line.find("max:") != std::string::npos) {
      parse_yaml_scalar(line, "max", out.constraints.n_max);
    }
    parse_yaml_scalar(line, "delta_rate", out.constraints.delta_rate_max);
    parse_yaml_scalar(line, "n_rate", out.constraints.n_rate_max);
  }

  out.encoder_onnx_path = model_dir + "/encoder.onnx";

  std::ifstream nxj(model_dir + "/norm_x.json");
  std::ifstream nuj(model_dir + "/norm_u.json");
  if (!nxj || !nuj) return false;
  std::stringstream xss, uss;
  xss << nxj.rdbuf();
  uss << nuj.rdbuf();
  const std::string xtxt = xss.str();
  const std::string utxt = uss.str();
  std::vector<double> mu, sigma, mu_u, sigma_u;
  if (!parse_json_array(xtxt, "mu", mu) || !parse_json_array(xtxt, "sigma", sigma)) return false;
  if (!parse_json_array(utxt, "mu", mu_u) || !parse_json_array(utxt, "sigma", sigma_u)) return false;
  if (mu.size() != NX || sigma.size() != NX || mu_u.size() != NU || sigma_u.size() != NU) return false;
  for (int i = 0; i < NX; ++i) {
    out.norm.mu[i] = mu[i];
    out.norm.sigma[i] = std::max(sigma[i], 1e-6);
  }
  for (int i = 0; i < NU; ++i) {
    out.norm.mu_u[i] = mu_u[i];
    out.norm.sigma_u[i] = std::max(sigma_u[i], 1e-6);
  }

  out.mats.nz = out.nz;
  out.mats.nu = NU;
  const std::size_t a_sz = static_cast<std::size_t>(out.nz) * static_cast<std::size_t>(out.nz);
  const std::size_t b_sz = static_cast<std::size_t>(out.nz) * static_cast<std::size_t>(NU);
  out.mats.A = read_f64_bin(model_dir + "/A.bin", a_sz);
  out.mats.B = read_f64_bin(model_dir + "/B.bin", b_sz);

  const std::string cx_path = model_dir + "/Cx.bin";
  std::ifstream cx_probe(cx_path);
  if (cx_probe.good()) {
    out.mats.Cx = read_f64_bin(cx_path, static_cast<std::size_t>(NX) * static_cast<std::size_t>(out.nz));
    out.mats.has_cx = true;
    std::ifstream bias(model_dir + "/decoder_bias.json");
    if (bias) {
      std::stringstream bss;
      bss << bias.rdbuf();
      std::vector<double> bvec;
      if (parse_json_array(bss.str(), "bias", bvec) && bvec.size() == NX) {
        for (int i = 0; i < NX; ++i) out.mats.decoder_bias[i] = bvec[i];
      }
    }
  }

  return true;
}

}  // namespace vessel_control
