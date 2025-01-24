#include <string>

#include "Kernel.h"

extern const char _binary_kernels_gemm_kernel_int1_cu_start,
    _binary_kernels_gemm_kernel_int1_cu_end;

namespace ccglib::mma {

template <>
Kernel::Parameters Kernel::GetCompileParameters<ValueType::int1>() const {
  return Kernel::Parameters{.m_per_block = 64,
                            .m_per_warp = 32,
                            .m_per_wmma = 16,

                            .n_per_block = 64,
                            .n_per_warp = 32,
                            .n_per_wmma = 8,

                            .k_per_wmma = 256,
                            .nbuffer = 4};
}

template <> std::string Kernel::GetSource<ValueType::int1>() const {
  return std::string(&_binary_kernels_gemm_kernel_int1_cu_start,
                     &_binary_kernels_gemm_kernel_int1_cu_end);
}
} // namespace ccglib::mma