#include <stdexcept>

#include "Kernel.h"

namespace ccglib::mma {

Kernel::Kernel(Precision precision, Variant variant)
    : precision_(precision), variant_(variant) {
  SetParameters(precision);
};

dim3 Kernel::GetThreads() const {
  return dim3(parameters_.warp_size,
              parameters_.n_per_block / parameters_.n_per_warp,
              parameters_.m_per_block / parameters_.m_per_warp);
}

void Kernel::SetParameters(Precision precision) {
  switch (precision) {
  case ccglib::mma::Precision::float16:
    parameters_ = Kernel::GetParameters<Precision::float16>();
    break;
  case ccglib::mma::Precision::float32:
    parameters_ = Kernel::GetParameters<Precision::float32>();
    break;
  case ccglib::mma::Precision::int1:
    parameters_ = Kernel::GetParameters<Precision::int1>();
    break;
  default:
    throw std::runtime_error(
        "No kernel parameters available for selected precision");
  }
}

std::string Kernel::GetSource() const {
  // precision determines in which file the kernel resides
  switch (precision_) {
  case ccglib::mma::float16:
    return Kernel::GetSource<Precision::float16>();
  case ccglib::mma::float32:
    return Kernel::GetSource<Precision::float32>();
  case ccglib::mma::int1:
    return Kernel::GetSource<Precision::int1>();
  default:
    throw std::runtime_error("Selected GEMM precision is not available");
  }
}

std::string Kernel::GetName() const {
  switch (variant_) {
  case ccglib::mma::basic:
    return "wmma_complex_gemm_basic";
  case ccglib::mma::opt:
    return "wmma_complex_gemm_opt";
  default:
    throw std::runtime_error("Selected GEMM variant is not available");
  }
}

} // end namespace ccglib::mma
