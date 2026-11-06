#ifndef WMMA_EXTENSION_H_
#define WMMA_EXTENSION_H_

/*
There are PTX mma instructions for 16x8x256 mma in binary precision, but these
are missing from the wmma interface
This header adds the missing code

Based on John Romein's 1 bit mma code
*/

inline __device__ unsigned laneid() {
  unsigned laneid;

  asm("mov.u32 %0, %%laneid;" : "=r"(laneid));
  return laneid;
}

namespace nvcuda {
namespace wmma {

#if defined(__HIP_PLATFORM_AMD__)
// AMD HIP platform: enable AMD-specific fragments
// (add AMD-specific fragments here if needed)

#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) ||       \
    defined(__GFX12__)

#endif

#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 730)
// CUDA platform, architecture >= 730: enable CUDA fragments

// 16x8x256, a row major, b col major, binary precision
template <>
class fragment<matrix_a, 16, 8, 256, experimental::precision::b1, row_major>
    : public __frag_base<experimental::precision::b1, 32, 4> {};
template <>
class fragment<matrix_b, 16, 8, 256, experimental::precision::b1, col_major>
    : public __frag_base<experimental::precision::b1, 32, 2> {};

#if (__CUDA_ARCH__ >= 890)
template <>
class fragment<matrix_a, 16, 8, 32, fp8_e4m3, row_major>
    : public __frag_base<int, 4> {};
template <>
class fragment<matrix_b, 16, 8, 32, fp8_e4m3, col_major>
    : public __frag_base<int, 2> {};
template <>
class fragment<matrix_a, 16, 8, 32, fp8_e5m2, row_major>
    : public __frag_base<int, 4> {};
template <>
class fragment<matrix_b, 16, 8, 32, fp8_e5m2, col_major>
    : public __frag_base<int, 2> {};
#endif // __CUDA_ARCH__ >= 890

// Fragments for the warp-level mma.sync instructions with kind::f8f6f4
// (consumer/workstation Blackwell: sm_120a and sm_121a). FP4 and FP6 values
// are byte-padded like FP8, so the fragment shapes and the per-thread
// fragment layout match the FP8 fragments. The instruction was verified on
// an NVIDIA RTX PRO 6000 Blackwell (sm_120).
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
template <>
class fragment<matrix_a, 16, 8, 32, fp4_e2m1, row_major>
    : public __frag_base<int, 4> {};
template <>
class fragment<matrix_b, 16, 8, 32, fp4_e2m1, col_major>
    : public __frag_base<int, 2> {};
template <>
class fragment<matrix_a, 16, 8, 32, fp6_e2m3, row_major>
    : public __frag_base<int, 4> {};
template <>
class fragment<matrix_b, 16, 8, 32, fp6_e2m3, col_major>
    : public __frag_base<int, 2> {};
template <>
class fragment<matrix_a, 16, 8, 32, fp6_e3m2, row_major>
    : public __frag_base<int, 4> {};
template <>
class fragment<matrix_b, 16, 8, 32, fp6_e3m2, col_major>
    : public __frag_base<int, 2> {};
#endif

template <>
class fragment<accumulator, 16, 8, 32, float> : public __frag_base<float, 4> {};
template <>
class fragment<accumulator, 16, 8, 256, int> : public __frag_base<int, 4> {};

inline __device__ void bmma_sync(
    fragment<accumulator, 16, 8, 256, int> &d,
    const fragment<matrix_a, 16, 8, 256, experimental::precision::b1, row_major>
        &a,
    const fragment<matrix_b, 16, 8, 256, experimental::precision::b1, col_major>
        &b,
    const fragment<accumulator, 16, 8, 256, int> &c,
    experimental::bmmaBitOp bitOp = experimental::bmmaBitOpXOR,
    experimental::bmmaAccumulateOp = experimental::bmmaAccumulateOpPOPC) {

  if (bitOp == experimental::bmmaBitOpXOR) {
    asm("mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc {%0, %1, "
        "%2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
        : "=r"(d.x[0]), "=r"(d.x[1]), "=r"(d.x[2]), "=r"(d.x[3])
        : "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]), "r"(b.x[0]),
          "r"(b.x[1]), "r"(c.x[0]), "r"(c.x[1]), "r"(c.x[2]), "r"(c.x[3]));
  } else if (bitOp == experimental::bmmaBitOpAND) {
    asm("mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc {%0, %1, "
        "%2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
        : "=r"(d.x[0]), "=r"(d.x[1]), "=r"(d.x[2]), "=r"(d.x[3])
        : "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]), "r"(b.x[0]),
          "r"(b.x[1]), "r"(c.x[0]), "r"(c.x[1]), "r"(c.x[2]), "r"(c.x[3]));
  }
}

#if __CUDA_ARCH__ >= 890
inline __device__ void
mma_sync(fragment<accumulator, 16, 8, 32, float> &d,
         const fragment<matrix_a, 16, 8, 32, fp8_e4m3, row_major> &a,
         const fragment<matrix_b, 16, 8, 32, fp8_e4m3, col_major> &b,
         const fragment<accumulator, 16, 8, 32, float> &c) {
  asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, "
      "{%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
      : "=f"(d.x[0]), "=f"(d.x[1]), "=f"(d.x[2]), "=f"(d.x[3])
      : "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]), "r"(b.x[0]),
        "r"(b.x[1]), "f"(c.x[0]), "f"(c.x[1]), "f"(c.x[2]), "f"(c.x[3]));
}

inline __device__ void
load_matrix_sync(fragment<matrix_a, 16, 8, 32, fp8_e4m3, row_major> &a,
                 const void *p, size_t ldm) {
  a.x[0] = ((const int *)p)[ldm / 4 * (laneid() / 4) + laneid() % 4];
  a.x[1] = ((const int *)p)[ldm / 4 * (laneid() / 4 + 8) + laneid() % 4];
  a.x[2] = ((const int *)p)[ldm / 4 * (laneid() / 4) + laneid() % 4 + 4];
  a.x[3] = ((const int *)p)[ldm / 4 * (laneid() / 4 + 8) + laneid() % 4 + 4];
}

inline __device__ void
load_matrix_sync(fragment<matrix_b, 16, 8, 32, fp8_e4m3, col_major> &b,
                 const void *p, size_t ldm) {
  b.x[0] = ((const int *)p)[ldm / 4 * (laneid() / 4) + laneid() % 4];
  b.x[1] = ((const int *)p)[ldm / 4 * (laneid() / 4) + laneid() % 4 + 4];
}

inline __device__ void
mma_sync(fragment<accumulator, 16, 8, 32, float> &d,
         const fragment<matrix_a, 16, 8, 32, fp8_e5m2, row_major> &a,
         const fragment<matrix_b, 16, 8, 32, fp8_e5m2, col_major> &b,
         const fragment<accumulator, 16, 8, 32, float> &c) {
  asm("mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 {%0,%1,%2,%3}, "
      "{%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
      : "=f"(d.x[0]), "=f"(d.x[1]), "=f"(d.x[2]), "=f"(d.x[3])
      : "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]), "r"(b.x[0]),
        "r"(b.x[1]), "f"(c.x[0]), "f"(c.x[1]), "f"(c.x[2]), "f"(c.x[3]));
}

inline __device__ void
load_matrix_sync(fragment<matrix_a, 16, 8, 32, fp8_e5m2, row_major> &a,
                 const void *p, size_t ldm) {
  a.x[0] = ((const int *)p)[ldm / 4 * (laneid() / 4) + laneid() % 4];
  a.x[1] = ((const int *)p)[ldm / 4 * (laneid() / 4 + 8) + laneid() % 4];
  a.x[2] = ((const int *)p)[ldm / 4 * (laneid() / 4) + laneid() % 4 + 4];
  a.x[3] = ((const int *)p)[ldm / 4 * (laneid() / 4 + 8) + laneid() % 4 + 4];
}

inline __device__ void
load_matrix_sync(fragment<matrix_b, 16, 8, 32, fp8_e5m2, col_major> &b,
                 const void *p, size_t ldm) {
  b.x[0] = ((const int *)p)[ldm / 4 * (laneid() / 4) + laneid() % 4];
  b.x[1] = ((const int *)p)[ldm / 4 * (laneid() / 4) + laneid() % 4 + 4];
}
#endif

#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
// mma.sync and load_matrix_sync for the byte-padded FP4 and FP6 types. The
// per-thread fragment layout is identical to the FP8 layout because every
// value occupies one byte. Values of matrix B are stored transposed and
// k-major, matching the row.col operand layout of the instruction.
inline __device__ void
mma_sync(fragment<accumulator, 16, 8, 32, float> &d,
         const fragment<matrix_a, 16, 8, 32, fp4_e2m1, row_major> &a,
         const fragment<matrix_b, 16, 8, 32, fp4_e2m1, col_major> &b,
         const fragment<accumulator, 16, 8, 32, float> &c) {
  asm("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
      : "=f"(d.x[0]), "=f"(d.x[1]), "=f"(d.x[2]), "=f"(d.x[3])
      : "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]), "r"(b.x[0]),
        "r"(b.x[1]), "f"(c.x[0]), "f"(c.x[1]), "f"(c.x[2]), "f"(c.x[3]));
}

inline __device__ void
mma_sync(fragment<accumulator, 16, 8, 32, float> &d,
         const fragment<matrix_a, 16, 8, 32, fp6_e2m3, row_major> &a,
         const fragment<matrix_b, 16, 8, 32, fp6_e2m3, col_major> &b,
         const fragment<accumulator, 16, 8, 32, float> &c) {
  asm("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m3.e2m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
      : "=f"(d.x[0]), "=f"(d.x[1]), "=f"(d.x[2]), "=f"(d.x[3])
      : "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]), "r"(b.x[0]),
        "r"(b.x[1]), "f"(c.x[0]), "f"(c.x[1]), "f"(c.x[2]), "f"(c.x[3]));
}

inline __device__ void
mma_sync(fragment<accumulator, 16, 8, 32, float> &d,
         const fragment<matrix_a, 16, 8, 32, fp6_e3m2, row_major> &a,
         const fragment<matrix_b, 16, 8, 32, fp6_e3m2, col_major> &b,
         const fragment<accumulator, 16, 8, 32, float> &c) {
  asm("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e3m2.e3m2.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
      : "=f"(d.x[0]), "=f"(d.x[1]), "=f"(d.x[2]), "=f"(d.x[3])
      : "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]), "r"(b.x[0]),
        "r"(b.x[1]), "f"(c.x[0]), "f"(c.x[1]), "f"(c.x[2]), "f"(c.x[3]));
}

// The load_matrix_sync overloads below read fragments directly from
// (shared or global) memory with byte-padded elements. The memory layout and
// the per-thread mapping are identical to FP8, so one implementation covers
// all six fragment types.
template <typename Fragment>
inline __device__ void load_matrix_sync_f8f6f4_a(Fragment &a, const void *p,
                                                 size_t ldm) {
  a.x[0] = ((const int *)p)[ldm / 4 * (laneid() / 4) + laneid() % 4];
  a.x[1] = ((const int *)p)[ldm / 4 * (laneid() / 4 + 8) + laneid() % 4];
  a.x[2] = ((const int *)p)[ldm / 4 * (laneid() / 4) + laneid() % 4 + 4];
  a.x[3] = ((const int *)p)[ldm / 4 * (laneid() / 4 + 8) + laneid() % 4 + 4];
}

template <typename Fragment>
inline __device__ void load_matrix_sync_f8f6f4_b(Fragment &b, const void *p,
                                                 size_t ldm) {
  b.x[0] = ((const int *)p)[ldm / 4 * (laneid() / 4) + laneid() % 4];
  b.x[1] = ((const int *)p)[ldm / 4 * (laneid() / 4) + laneid() % 4 + 4];
}

inline __device__ void
load_matrix_sync(fragment<matrix_a, 16, 8, 32, fp4_e2m1, row_major> &a,
                 const void *p, size_t ldm) {
  load_matrix_sync_f8f6f4_a(a, p, ldm);
}

inline __device__ void
load_matrix_sync(fragment<matrix_b, 16, 8, 32, fp4_e2m1, col_major> &b,
                 const void *p, size_t ldm) {
  load_matrix_sync_f8f6f4_b(b, p, ldm);
}

inline __device__ void
load_matrix_sync(fragment<matrix_a, 16, 8, 32, fp6_e2m3, row_major> &a,
                 const void *p, size_t ldm) {
  load_matrix_sync_f8f6f4_a(a, p, ldm);
}

inline __device__ void
load_matrix_sync(fragment<matrix_b, 16, 8, 32, fp6_e2m3, col_major> &b,
                 const void *p, size_t ldm) {
  load_matrix_sync_f8f6f4_b(b, p, ldm);
}

inline __device__ void
load_matrix_sync(fragment<matrix_a, 16, 8, 32, fp6_e3m2, row_major> &a,
                 const void *p, size_t ldm) {
  load_matrix_sync_f8f6f4_a(a, p, ldm);
}

inline __device__ void
load_matrix_sync(fragment<matrix_b, 16, 8, 32, fp6_e3m2, col_major> &b,
                 const void *p, size_t ldm) {
  load_matrix_sync_f8f6f4_b(b, p, ldm);
}
#endif // __CUDA_ARCH_FEAT_SM120_ALL || __CUDA_ARCH_FEAT_SM121_ALL

inline __device__ void load_matrix_sync(
    fragment<matrix_a, 16, 8, 256, experimental::precision::b1, row_major> &a,
    const void *p, size_t ldm) {
  a.x[0] = ((const int *)p)[ldm / 32 * (laneid() / 4) + laneid() % 4];
  a.x[1] = ((const int *)p)[ldm / 32 * (laneid() / 4 + 8) + laneid() % 4];
  a.x[2] = ((const int *)p)[ldm / 32 * (laneid() / 4) + laneid() % 4 + 4];
  a.x[3] = ((const int *)p)[ldm / 32 * (laneid() / 4 + 8) + laneid() % 4 + 4];
}

inline __device__ void load_matrix_sync(
    fragment<matrix_b, 16, 8, 256, experimental::precision::b1, col_major> &b,
    const void *p, size_t ldm) {
  b.x[0] = ((const int *)p)[ldm / 32 * (laneid() / 4) + laneid() % 4];
  b.x[1] = ((const int *)p)[ldm / 32 * (laneid() / 4) + laneid() % 4 + 4];
}

inline __device__ void
store_matrix_sync(float *p, const fragment<accumulator, 16, 8, 32, float> &d,
                  unsigned ldm, layout_t layout) {
  if (layout == mem_row_major) {
    ((float2 *)p)[ldm / 2 * (laneid() / 4) + laneid() % 4] =
        make_float2(d.x[0], d.x[1]);
    ((float2 *)p)[ldm / 2 * (laneid() / 4 + 8) + laneid() % 4] =
        make_float2(d.x[2], d.x[3]);
  } else {
    p[(laneid() % 4) * 2 * ldm + (laneid() / 4)] = d.x[0];
    p[(laneid() % 4) * 2 * ldm + (laneid() / 4) + ldm] = d.x[1];
    p[(laneid() % 4) * 2 * ldm + (laneid() / 4 + 8)] = d.x[2];
    p[(laneid() % 4) * 2 * ldm + (laneid() / 4 + 8) + ldm] = d.x[3];
  }
}

inline __device__ void
store_matrix_sync(int *p, const fragment<accumulator, 16, 8, 256, int> &d,
                  size_t ldm, layout_t layout) {
  if (layout == mem_row_major) {
    ((int2 *)p)[ldm / 2 * (laneid() / 4) + laneid() % 4] =
        make_int2(d.x[0], d.x[1]);
    ((int2 *)p)[ldm / 2 * (laneid() / 4 + 8) + laneid() % 4] =
        make_int2(d.x[2], d.x[3]);
  } else {
    p[(laneid() % 4) * 2 * ldm + (laneid() / 4)] = d.x[0];
    p[(laneid() % 4) * 2 * ldm + (laneid() / 4) + ldm] = d.x[1];
    p[(laneid() % 4) * 2 * ldm + (laneid() / 4 + 8)] = d.x[2];
    p[(laneid() % 4) * 2 * ldm + (laneid() / 4 + 8) + ldm] = d.x[3];
  }
}

inline __device__ void
load_matrix_sync(fragment<accumulator, 16, 8, 256, int> &c, const void *p,
                 size_t ldm, layout_t layout) {
  if (layout == mem_row_major) {
    c.x[0] = ((int *)p)[ldm * (laneid() / 4) + (laneid() % 4) * 2];
    c.x[1] = ((int *)p)[ldm * (laneid() / 4) + (laneid() % 4) * 2 + 1];
    c.x[2] = ((int *)p)[ldm * (laneid() / 4 + 8) + (laneid() % 4) * 2];
    c.x[3] = ((int *)p)[ldm * (laneid() / 4 + 8) + (laneid() % 4) * 2 + 1];
  } else {
    c.x[0] = ((int *)p)[(laneid() % 4) * 2 * ldm + (laneid() / 4)];
    c.x[1] = ((int *)p)[(laneid() % 4) * 2 * ldm + (laneid() / 4) + ldm];
    c.x[2] = ((int *)p)[(laneid() % 4) * 2 * ldm + (laneid() / 4 + 8)];
    c.x[3] = ((int *)p)[(laneid() % 4) * 2 * ldm + (laneid() / 4 + 8 + ldm)];
  }
}
#endif

} // namespace wmma
} // namespace nvcuda

#endif // WMMA_EXTENSION_H_
