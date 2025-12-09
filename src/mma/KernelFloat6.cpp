#include <string>

#include "Kernel.h"

extern const char _binary_kernels_gemm_kernel_float_cu_start,
    _binary_kernels_gemm_kernel_float_cu_end;

namespace ccglib::mma {

template <>
Kernel::Parameters Kernel::GetCompileParameters<ValueType::float6e2m3>() const {
  // The preprocessor statements below force clang-format to use abnormal
  // format. To be consistent with the rest of the code, it is temporarily
  // disabled.
  // clang-format off
  return Kernel::Parameters{.m_per_block = 128,
                            .m_per_warp = 32,
                            .m_per_wmma = 16,

                            .n_per_block = 32,
                            .n_per_warp = 32,
                            .n_per_wmma = 8,

                            .k_per_wmma = 32,
#if defined(__HIP_PLATFORM_AMD__)
    .nbuffer = 1
#else
    .nbuffer = 4
#endif
  };
  // clang-format on
}

template <> std::string Kernel::GetSource<ValueType::float6e2m3>() const {
  return std::string(&_binary_kernels_gemm_kernel_float_cu_start,
                     &_binary_kernels_gemm_kernel_float_cu_end);
}

template <>
Kernel::Parameters Kernel::GetCompileParameters<ValueType::float6e3m2>() const {
  // The preprocessor statements below force clang-format to use abnormal
  // format. To be consistent with the rest of the code, it is temporarily
  // disabled.
  // clang-format off
  return Kernel::Parameters{.m_per_block = 128,
                            .m_per_warp = 32,
                            .m_per_wmma = 16,

                            .n_per_block = 32,
                            .n_per_warp = 32,
                            .n_per_wmma = 8,
                            
                            .k_per_wmma = 32,
#if defined(__HIP_PLATFORM_AMD__)
    .nbuffer = 1
#else
    .nbuffer = 4
#endif
  };
  // clang-format on
}

template <> std::string Kernel::GetSource<ValueType::float6e3m2>() const {
  return std::string(&_binary_kernels_gemm_kernel_float_cu_start,
                     &_binary_kernels_gemm_kernel_float_cu_end);
}
} // namespace ccglib::mma