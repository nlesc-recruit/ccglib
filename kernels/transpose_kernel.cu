#include <cuda/std/limits>
#include <cuda_fp16.h>

#ifndef COMPLEX
#define COMPLEX 2
#endif
#ifndef REAL
#define REAL 0
#endif
#ifndef IMAG
#define IMAG 1
#endif

#if NBIT == 16
using T = half;
#elif NBIT == 32
using T = float;
#elif NBIT == 1
using T = unsigned int;
#else
#error NBIT must be 1, 16, 32
#endif

#define PACKING_FACTOR (sizeof(T) * CHAR_BIT / NBIT)

extern "C" {
__global__ void transpose(
    T out[BATCH_SIZE][M_GLOBAL / M_CHUNK][N_GLOBAL / N_CHUNK][COMPLEX][M_CHUNK]
         [N_CHUNK / PACKING_FACTOR],
    const T in[BATCH_SIZE][COMPLEX][M_GLOBAL][N_GLOBAL / PACKING_FACTOR]) {
  const size_t idx_B = blockIdx.z;
  const size_t idx_N =
      threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
  const size_t idx_M =
      threadIdx.y + blockIdx.y * static_cast<size_t>(blockDim.y);

  static_assert(M_GLOBAL % M_CHUNK == 0);
  static_assert(N_GLOBAL % N_CHUNK == 0);
  static_assert(N_CHUNK % PACKING_FACTOR == 0);

  if (idx_B < BATCH_SIZE && idx_M < M_GLOBAL &&
      idx_N < N_GLOBAL / PACKING_FACTOR) {
    size_t b = idx_B;
    size_t m = idx_M / M_CHUNK;
    size_t m_c = idx_M % M_CHUNK;
    size_t n = idx_N / (N_CHUNK / PACKING_FACTOR);
    size_t n_c = idx_N % (N_CHUNK / PACKING_FACTOR);

    out[b][m][n][REAL][m_c][n_c] = in[b][REAL][idx_M][idx_N];
    out[b][m][n][IMAG][m_c][n_c] = in[b][IMAG][idx_M][idx_N];
  }
};
}
