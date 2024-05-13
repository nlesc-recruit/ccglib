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
#else
#error NBIT must be 16
#endif

extern "C" {
__global__ void
transpose(T out[B][M / M_CHUNK][N / N_CHUNK][COMPLEX][M_CHUNK][N_CHUNK],
          const T in[B][COMPLEX][M][N]) {
  const int idx_B = blockIdx.z;
  const int idx_N = threadIdx.x + blockDim.x * blockIdx.x;
  const int idx_M = threadIdx.y + blockDim.y * blockIdx.y;

  static_assert(M % M_CHUNK == 0);
  static_assert(N % N_CHUNK == 0);

  if (idx_B < B && idx_M < M && idx_N < N) {
    int b = idx_B;
    int m = idx_M / M_CHUNK;
    int m_c = idx_M % M_CHUNK;
    int n = idx_N / N_CHUNK;
    int n_c = idx_N % N_CHUNK;

    out[b][m][n][REAL][m_c][n_c] = in[b][REAL][idx_M][idx_N];
    out[b][m][n][IMAG][m_c][n_c] = in[b][IMAG][idx_M][idx_N];
  }
};
}
