#if defined(__HIP__)
#include <limits.h>
#else
#include <cuda/std/limits>
#endif

#include "ccglib/fp16.h"

#ifndef COMPLEX
#define COMPLEX 2
#endif
#ifndef REAL
#define REAL 0
#endif
#ifndef IMAG
#define IMAG 1
#endif

#if NBIT_IN == 16
using T = half;
#elif NBIT_IN == 32
using T = float;
#elif NBIT_IN == 1
using T = unsigned int;
#else
#error NBIT_IN must be 1, 16, 32
#endif

#define PACKING_FACTOR (sizeof(T) * CHAR_BIT / NBIT_IN)

#define M_IS_PADDED ((M_GLOBAL % M_CHUNK) != 0)
#define N_IS_PADDED ((N_GLOBAL % N_CHUNK) != 0)

#define M_GLOBAL_PADDED ((M_GLOBAL / M_CHUNK + M_IS_PADDED) * M_CHUNK)
#define N_GLOBAL_PADDED ((N_GLOBAL / N_CHUNK + N_IS_PADDED) * N_CHUNK)

#if defined(COMPLEX_MIDDLE)
using Input = T[BATCH_SIZE][COMPLEX][M_GLOBAL][N_GLOBAL / PACKING_FACTOR];
#elif defined(COMPLEX_LAST)
using Input = T[BATCH_SIZE][M_GLOBAL][N_GLOBAL / PACKING_FACTOR][COMPLEX];
#endif
using Output =
    T[BATCH_SIZE][M_GLOBAL_PADDED / M_CHUNK][N_GLOBAL_PADDED / N_CHUNK][COMPLEX]
     [M_CHUNK][N_CHUNK / PACKING_FACTOR];

extern "C" {
__global__ void transpose(Output out, const Input in) {
  const size_t idx_B = blockIdx.z;
  const size_t idx_N =
      threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
  const size_t idx_M =
      threadIdx.y + blockIdx.y * static_cast<size_t>(blockDim.y);

  static_assert(M_GLOBAL_PADDED % M_CHUNK == 0);
  static_assert(N_GLOBAL_PADDED % N_CHUNK == 0);
  static_assert(N_CHUNK % PACKING_FACTOR == 0);

  if (idx_B < BATCH_SIZE && idx_M < M_GLOBAL_PADDED &&
      idx_N < N_GLOBAL_PADDED / PACKING_FACTOR) {
    size_t b = idx_B;
    size_t m = idx_M / M_CHUNK;
    size_t m_c = idx_M % M_CHUNK;
    size_t n = idx_N / (N_CHUNK / PACKING_FACTOR);
    size_t n_c = idx_N % (N_CHUNK / PACKING_FACTOR);

    if (idx_M < M_GLOBAL && idx_N < N_GLOBAL / PACKING_FACTOR) {
#if defined(COMPLEX_MIDDLE)
      out[b][m][n][REAL][m_c][n_c] = in[b][REAL][idx_M][idx_N];
      out[b][m][n][IMAG][m_c][n_c] = in[b][IMAG][idx_M][idx_N];
#elif defined(COMPLEX_LAST)
      out[b][m][n][REAL][m_c][n_c] = in[b][idx_M][idx_N][REAL];
      out[b][m][n][IMAG][m_c][n_c] = in[b][idx_M][idx_N][IMAG];
#endif
    } else {
      out[b][m][n][REAL][m_c][n_c] = 0;
      out[b][m][n][IMAG][m_c][n_c] = 0;
    }
  }
};
}
