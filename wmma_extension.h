/*
There are PTX mma instructions for 16x8x256 mma in binary precision, but these
are missing from the wmma interface
This header adds the missing code

Based on John Romein's 1 bit mma code
*/

#include <mma.h>

inline __device__ unsigned laneid() {
  unsigned laneid;

  asm("mov.u32 %0, %%laneid;" : "=r"(laneid));
  return laneid;
}

namespace nvcuda {
namespace wmma {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 730

// 16x8x256, a row major, b col major, binary precision
template <>
class fragment<matrix_a, 16, 8, 256, experimental::precision::b1, row_major>
    : public __frag_base<experimental::precision::b1, 32, 4> {};
template <>
class fragment<matrix_b, 16, 8, 256, experimental::precision::b1, col_major>
    : public __frag_base<experimental::precision::b1, 32, 2> {};
template <>
class fragment<accumulator, 16, 8, 256, int> : public __frag_base<int, 4> {};

inline __device__ void bmma_sync(fragment<accumulator, 16, 8, 256, int>& d,
		     const fragment<matrix_a, 16, 8, 256, experimental::precision::b1, row_major>& a,
		     const fragment<matrix_b, 16, 8, 256, experimental::precision::b1, col_major>& b,
		     const fragment<accumulator, 16, 8, 256, int>& c /*,
		     experimental::bmmaBitOp = experimental::bmmaBitOpXOR,
		     experimental::bmmaAccumulateOp = experimental::bmmaAccumulateOpPOPC */) {
  asm("mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc {%0, %1, %2, "
      "%3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
      : "=r"(d.x[0]), "=r"(d.x[1]), "=r"(d.x[2]), "=r"(d.x[3])
      : "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]), "r"(b.x[0]),
        "r"(b.x[1]), "r"(c.x[0]), "r"(c.x[1]), "r"(c.x[2]), "r"(c.x[3]));
}

inline __device__ void load_matrix_sync(
    fragment<matrix_a, 16, 8, 256, experimental::precision::b1, row_major> &a,
    const void *p, unsigned ldm) {
  a.x[0] = ((const int *)p)[ldm / 32 * (laneid() / 4) + laneid() % 4];
  a.x[1] = ((const int *)p)[ldm / 32 * (laneid() / 4 + 8) + laneid() % 4];
  a.x[2] = ((const int *)p)[ldm / 32 * (laneid() / 4) + laneid() % 4 + 4];
  a.x[3] = ((const int *)p)[ldm / 32 * (laneid() / 4 + 8) + laneid() % 4 + 4];
}

inline __device__ void load_matrix_sync(
    fragment<matrix_b, 16, 8, 256, experimental::precision::b1, col_major> &b,
    const void *p, unsigned ldm) {
  b.x[0] = ((const int *)p)[ldm / 32 * (laneid() / 4) + laneid() % 4];
  b.x[1] = ((const int *)p)[ldm / 32 * (laneid() / 4) + laneid() % 4 + 4];
}

inline __device__ void
store_matrix_sync(int *p, const fragment<accumulator, 16, 8, 256, int> &d,
                  unsigned ldm, layout_t layout) {
  // FIXME: only row-major supported
  ((int2 *)p)[ldm / 2 * (laneid() / 4) + laneid() % 4] =
      make_int2(d.x[0], d.x[1]);
  ((int2 *)p)[ldm / 2 * (laneid() / 4 + 8) + laneid() % 4] =
      make_int2(d.x[2], d.x[3]);
}
#endif
} // namespace wmma
} // namespace nvcuda
