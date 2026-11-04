#ifndef CCGLIB_FP4_H_
#define CCGLIB_FP4_H_

// FP4 (E2M1) type for use as GEMM input.
//
// The warp-level tensor core instructions with kind::f8f6f4 (sm_120a and
// sm_121a) read FP4 values from an 8-bit container holding the 4-bit code in
// bits 5:2. This differs from the vendor type (__nv_fp4_e2m1 and
// __hip_fp4_e2m1), which stores the code in bits 3:0. This type wraps the
// code in the padded container so that buffers of fp4_e2m1 can be passed to
// the kernels unchanged, like all other input types. The layout of the
// container was verified on an NVIDIA RTX PRO 6000 Blackwell (sm_120).

#if defined(__HIP__)
#if HIP_VERSION_MAJOR < 7
// FP4 support was added in HIP 7.0, see tag history at
// https://github.com/ROCm/clr/blob/rocm-7.0.0/hipamd/include/hip/amd_detail/amd_hip_fp4.h
#define CCGLIB_HIP_NO_FP4
#else
#include <hip/hip_fp4.h>
#endif
#else
#include <cuda_fp4.h>
#endif

struct fp4_e2m1 {
  // Byte-padded container: bits 5:2 hold the E2M1 code, all other bits are 0.
  unsigned char x = 0;

  __host__ __device__ fp4_e2m1() = default;

  __host__ __device__ explicit fp4_e2m1(float value) : x(encode(value)) {}

  __host__ __device__ explicit operator float() const { return decode(x); }

  __host__ __device__ explicit operator double() const {
    return static_cast<double>(float(*this));
  }

  // Encode a float into the byte-padded container, rounding to nearest.
  __host__ __device__ static unsigned char encode(float value) {
#if defined(CCGLIB_HIP_NO_FP4)
    static_cast<void>(value);
    return 0;
#elif defined(__HIP__)
    return static_cast<unsigned char>(
        __hip_cvt_float_to_fp4(value, __HIP_E2M1, hipSatfinite) << 2);
#else
    return static_cast<unsigned char>(
        __nv_cvt_float_to_fp4(value, __NV_E2M1, cudaRoundNearest) << 2);
#endif
  }

  // Decode the byte-padded container back to a float.
  __host__ __device__ static float decode(unsigned char container) {
#if defined(CCGLIB_HIP_NO_FP4)
    static_cast<void>(container);
    return 0.0f;
#elif defined(__HIP__)
    return static_cast<float>(
        __hip_cvt_fp4_to_halfraw(container >> 2, __HIP_E2M1));
#else
    return __half2float(__nv_cvt_fp4_to_halfraw(container >> 2, __NV_E2M1));
#endif
  }
};

static_assert(sizeof(fp4_e2m1) == 1, "fp4_e2m1 must be one byte");

#endif // CCGLIB_FP4_H_
