#if defined(__HIP__)
#if HIP_VERSION_MAJOR < 7
// FP4 support was added in HIP 7.0, see tag history at
// https://github.com/ROCm/clr/blob/rocm-7.0.0/hipamd/include/hip/amd_detail/amd_hip_fp4.h

#ifndef CCGLIB_HIP_MOCK_FP4_TYPE
#define CCGLIB_HIP_MOCK_FP4_TYPE

// Helper to define mock HIP FP4 types
#define CCGLIB_HIP_DEFINE_MOCK_FP4_TYPE(TYPE_NAME)                             \
  struct __hip_##TYPE_NAME {                                                   \
    char x;                                                                    \
    __host__ __device__ __hip_##TYPE_NAME() : x(0) {}                          \
    __host__ __device__ explicit __hip_##TYPE_NAME(float) {}                   \
    __host__ __device__ explicit operator float() const { return 0.0f; }       \
  }

CCGLIB_HIP_DEFINE_MOCK_FP4_TYPE(fp4_e2m1);

#undef CCGLIB_HIP_DEFINE_MOCK_FP4_TYPE

#endif // CCGLIB_HIP_MOCK_FP4_TYPE
#endif

using fp4_e2m1 = __hip_fp4_e2m1;

#else
#include <cuda_fp4.h>
using fp4_e2m1 = __nv_fp4_e2m1;
#endif
