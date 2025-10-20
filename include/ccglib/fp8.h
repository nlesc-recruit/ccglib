#if defined(__HIP__)
#if HIP_VERSION_MAJOR < 6 || (HIP_VERSION_MAJOR == 6 && HIP_VERSION_MINOR < 2)
// FP8 support was added in HIP 6.2, see tag history at
// https://github.com/ROCm/clr/blob/rocm-6.2.1/hipamd/include/hip/amd_detail/amd_hip_fp8.h

#ifndef CCGLIB_HIP_MOCK_FP8_TYPE
#define CCGLIB_HIP_MOCK_FP8_TYPE

// Helper to define mock HIP FP8 types
#define CCGLIB_HIP_DEFINE_MOCK_FP8_TYPE(TYPE_NAME)                             \
  struct __hip_##TYPE_NAME {                                                   \
    char x;                                                                    \
    __host__ __device__ __hip_##TYPE_NAME() : x(0) {}                          \
    __host__ __device__ explicit __hip_##TYPE_NAME(float) {}                   \
    __host__ __device__ explicit operator float() const { return 0.0f; }       \
  }

CCGLIB_HIP_DEFINE_MOCK_FP8_TYPE(fp8_e5m2_fnuz);
CCGLIB_HIP_DEFINE_MOCK_FP8_TYPE(fp8_e4m3_fnuz);

#undef CCGLIB_HIP_DEFINE_MOCK_FP8_TYPE

#endif // CCGLIB_HIP_MOCK_FP8_TYPE
using fp8_e5m2 = __hip_fp8_e5m3_fnuz;
using fp8_e4m3 = __hip_fp8_e4m3_fnuz;
#else

#include <hip/hip_fp8.h>
#if HIP_VERSION_MAJOR == 6 && HIP_VERSION_MINOR == 2
// In HIP 6.2, the __hip_fp8_e4m3_fnuz is defined in hip_fp8.h
using fp8_e4m3 = __hip_fp8_e4m3_fnuz;
#else
// Since HIP 6.3, the HIP_FP8_TYPE_FNUZ macro can be used probed for
// availability of FP8
#if HIP_FP8_TYPE_OCP
using fp8_e5m2 = __hip_fp8_e5m2;
using fp8_e4m3 = __hip_fp8_e4m3;
#elif HIP_FP8_TYPE_FNUZ
using fp8_e5m2 = __hip_fp8_e5m2_fnuz;
using fp8_e4m3 = __hip_fp8_e4m3_fnuz;
#else
// Either HIP_FP8_TYPE_OCP or HIP_FP8_TYPE_FNUZ should be set.
// https://github.com/ROCm/clr/blob/rocm-6.3.0/hipamd/include/hip/amd_detail/amd_hip_fp8.h#L49
#error                                                                         \
    "Unexpected, ether HIP_FP8_TYPE_OCP or HIP_FP8_TYPE_FNUZ should be defined"
#endif
#endif
#endif
#else
#include <cuda_fp8.h>
using fp8_e5m2 = __nv_fp8_e5m2;
using fp8_e4m3 = __nv_fp8_e4m3;
#endif
