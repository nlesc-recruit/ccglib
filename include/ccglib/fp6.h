#if defined(__HIP__)
#if HIP_VERSION_MAJOR < 7
// FP6 support was added in HIP 7.0, see tag history at
// https://github.com/ROCm/clr/blob/rocm-7.0.0/hipamd/include/hip/amd_detail/amd_hip_fp6.h

#ifndef CCGLIB_HIP_MOCK_FP6_TYPE
#define CCGLIB_HIP_MOCK_FP6_TYPE

// Helper to define mock HIP FP6 types
#define CCGLIB_HIP_DEFINE_MOCK_FP6_TYPE(TYPE_NAME)                             \
  struct __hip_##TYPE_NAME {                                                   \
    char x;                                                                    \
    __host__ __device__ __hip_##TYPE_NAME() : x(0) {}                          \
    __host__ __device__ explicit __hip_##TYPE_NAME(float) {}                   \
    __host__ __device__ explicit operator float() const { return 0.0f; }       \
  }

CCGLIB_HIP_DEFINE_MOCK_FP6_TYPE(fp6_e2m3);
CCGLIB_HIP_DEFINE_MOCK_FP6_TYPE(fp6_e3m2);

#undef CCGLIB_HIP_DEFINE_MOCK_FP6_TYPE

#endif // CCGLIB_HIP_MOCK_FP6_TYPE
#endif

using fp6_e2m3 = __hip_fp6_e2m3;
using fp6_e3m2 = __hip_fp6_e3m2;

#else
#include <cuda_fp6.h>
using fp6_e2m3 = __nv_fp6_e2m3;
using fp6_e3m2 = __nv_fp6_e3m2;
#endif
