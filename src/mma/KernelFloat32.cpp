#include <string>

#include "Kernel.h"

extern const char _binary_kernels_gemm_kernel_float_cu_start,
    _binary_kernels_gemm_kernel_float_cu_end;

namespace ccglib::mma {

template <>
Kernel::Parameters Kernel::GetCompileParameters<ValueType::float32>() const {
  // clang-format off
  return Kernel::Parameters{.m_per_block = 128,
                            .m_per_warp = 128,
                            .m_per_wmma = 16,

                            .n_per_block = 64,
                            .n_per_warp = 16,
                            .n_per_wmma = 16,

                            .k_per_wmma = 8,
#if defined(__HIP_PLATFORM_AMD__)
      .nbuffer = 1
#else
      .nbuffer = 4
#endif
  };
  // clang-format on
}

template <> std::string Kernel::GetSource<ValueType::float32>() const {
  return std::string(&_binary_kernels_gemm_kernel_float_cu_start,
                     &_binary_kernels_gemm_kernel_float_cu_end);
}
} // namespace ccglib::mma