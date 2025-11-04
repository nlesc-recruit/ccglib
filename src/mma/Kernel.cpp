#include <stdexcept>

#include "Kernel.h"

namespace ccglib::mma {

Kernel::Kernel(Precision precision, const Variant &variant)
    : precision_(precision), variant_(variant) {
  SetParameters(precision);
};

dim3 Kernel::GetThreads(cu::Device &device) const {
  const unsigned warp_size = device.getAttribute(CU_DEVICE_ATTRIBUTE_WARP_SIZE);
  return dim3(warp_size, parameters_.n_per_block / parameters_.n_per_warp,
              parameters_.m_per_block / parameters_.m_per_warp);
}

void Kernel::SetParameters(Precision precision) {
  switch (precision.input_type) {
  case ValueType::float8e4m3:
    parameters_ = Kernel::GetCompileParameters<ValueType::float8e4m3>();
    break;
  case ValueType::float8e5m2:
    parameters_ = Kernel::GetCompileParameters<ValueType::float8e5m2>();
    break;
  case ValueType::float16:
    parameters_ = Kernel::GetCompileParameters<ValueType::float16>();
    break;
  case ValueType::bfloat16:
    parameters_ = Kernel::GetCompileParameters<ValueType::bfloat16>();
    break;
  case ValueType::float32:
    parameters_ = Kernel::GetCompileParameters<ValueType::float32>();
    break;
  case ValueType::int1:
    parameters_ = Kernel::GetCompileParameters<ValueType::int1>();
    break;
  default:
    throw std::runtime_error(
        "No kernel parameters available for selected precision");
  }
}

std::string Kernel::GetSource() const {
  // precision determines in which file the kernel resides
  switch (precision_.input_type) {
  case ValueType::float8e4m3:
    return Kernel::GetSource<ValueType::float8e4m3>();
  case ValueType::float8e5m2:
    return Kernel::GetSource<ValueType::float8e5m2>();
  case ValueType::float16:
    return Kernel::GetSource<ValueType::float16>();
  case ValueType::bfloat16:
    return Kernel::GetSource<ValueType::bfloat16>();
  case ValueType::float32:
    return Kernel::GetSource<ValueType::float32>();
  case ValueType::int1:
    return Kernel::GetSource<ValueType::int1>();
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
