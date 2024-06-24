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
#elif NBIT == 1
using T = unsigned int;
#else
#error NBIT must be 16 or 1
#endif

#define PACKING_FACTOR (sizeof(T) * CHAR_BIT / NBIT)

extern "C" {
__global__ void transpose(T out[B][M / M_CHUNK][N / N_CHUNK][COMPLEX][M_CHUNK]
                               [N_CHUNK / PACKING_FACTOR],
                          const T in[B][COMPLEX][M][N / PACKING_FACTOR]) {
  const int idx_B = blockIdx.z;
  const int idx_N = threadIdx.x + blockDim.x * blockIdx.x;
  const int idx_M = threadIdx.y + blockDim.y * blockIdx.y;

  static_assert(M % M_CHUNK == 0);
  static_assert(N % N_CHUNK == 0);
  static_assert(N_CHUNK % PACKING_FACTOR == 0);

  if (idx_B < B && idx_M < M && idx_N < N / PACKING_FACTOR) {
    int b = idx_B;
    int m = idx_M / M_CHUNK;
    int m_c = idx_M % M_CHUNK;
    int n = idx_N / (N_CHUNK / PACKING_FACTOR);
    int n_c = idx_N % (N_CHUNK / PACKING_FACTOR);

    out[b][m][n][REAL][m_c][n_c] = in[b][REAL][idx_M][idx_N];
    out[b][m][n][IMAG][m_c][n_c] = in[b][IMAG][idx_M][idx_N];
  }
};
}
