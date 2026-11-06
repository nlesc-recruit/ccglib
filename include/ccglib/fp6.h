#ifndef CCGLIB_FP6_H_
#define CCGLIB_FP6_H_

// FP6 (E2M3 and E3M2) types for use as GEMM inputs.
//
// The warp-level tensor core instructions with kind::f8f6f4 (sm_120a and
// sm_121a) read FP6 values from an 8-bit container holding the 6-bit code in
// bits 5:0, which matches the storage layout of the vendor types. Those types
// can therefore be used directly, like the FP8 types. The layout was verified
// on an NVIDIA RTX PRO 6000 Blackwell (sm_120).

#if defined(__HIP__)
#if HIP_VERSION_MAJOR < 7
// FP6 support was added in HIP 7.0, see tag history at
// https://github.com/ROCm/clr/blob/rocm-7.0.0/hipamd/include/hip/amd_detail/amd_hip_fp6.h

#ifndef CCGLIB_MOCK_FP6_TYPE
#define CCGLIB_MOCK_FP6_TYPE

// Helper to define mock HIP FP6 types for toolchains without FP6 support.
// Like the FP8 mocks, they carry no values.
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

#endif // CCGLIB_MOCK_FP6_TYPE
#else
#include <hip/hip_fp6.h>
#endif

using fp6_e2m3 = __hip_fp6_e2m3;
using fp6_e3m2 = __hip_fp6_e3m2;

#else
// cuda_runtime_api.h defines CUDART_VERSION and the __host__/__device__
// annotations and exists on all CUDA versions.
#include <cuda_runtime_api.h>
#if CUDART_VERSION < 12080
// The FP6 intrinsics were added in CUDA 12.8, see
// https://docs.nvidia.com/cuda/cuda-release-notes/

#ifndef CCGLIB_MOCK_FP6_TYPE
#define CCGLIB_MOCK_FP6_TYPE

// Helper to define mock CUDA FP6 types for toolchains without FP6 support.
// Like the FP8 mocks, they carry no values.
#define CCGLIB_CUDA_DEFINE_MOCK_FP6_TYPE(TYPE_NAME)                            \
  struct __nv_##TYPE_NAME {                                                    \
    char x;                                                                    \
    __host__ __device__ __nv_##TYPE_NAME() : x(0) {}                           \
    __host__ __device__ explicit __nv_##TYPE_NAME(float) {}                    \
    __host__ __device__ explicit operator float() const { return 0.0f; }       \
  }

CCGLIB_CUDA_DEFINE_MOCK_FP6_TYPE(fp6_e2m3);
CCGLIB_CUDA_DEFINE_MOCK_FP6_TYPE(fp6_e3m2);

#undef CCGLIB_CUDA_DEFINE_MOCK_FP6_TYPE

#endif // CCGLIB_MOCK_FP6_TYPE
#else
#include <cuda_fp6.h>
#endif

using fp6_e2m3 = __nv_fp6_e2m3;
using fp6_e3m2 = __nv_fp6_e3m2;

#endif

#endif // CCGLIB_FP6_H_
