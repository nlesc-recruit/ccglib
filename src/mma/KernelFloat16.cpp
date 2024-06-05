#include <string>

#include <cuda_runtime.h>

#include "Kernel.h"

extern const char _binary_kernels_gemm_kernel_float16_cu_start,
    _binary_kernels_gemm_kernel_float16_cu_end;

namespace ccglib::mma {

template <>
Kernel::Parameters Kernel::GetParameters<Precision::float16>() const {
  Kernel::Parameters kernel_parameters = {.m_per_block = 128,
                                          .m_per_warp = 32,
                                          .m_per_wmma = 16,

                                          .n_per_block = 64,
                                          .n_per_warp = 32,
                                          .n_per_wmma = 16,

                                          .k_per_wmma = 16,

                                          .warp_size = 32,
                                          .nbuffer = 4};

  return kernel_parameters;
}

template <> std::string Kernel::GetSource<Precision::float16>() const {
  return std::string(&_binary_kernels_gemm_kernel_float16_cu_start,
                     &_binary_kernels_gemm_kernel_float16_cu_end);
}
} // namespace ccglib::mma