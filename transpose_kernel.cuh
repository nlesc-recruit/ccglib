#ifndef TRANSPOSER_KERNEL_CUH
#define TRANSPOSER_KERNEL_CUH

#ifndef COMPLEX
#define COMPLEX 2
#endif
#ifndef REAL
#define REAL 0
#endif
#ifndef IMAG
#define IMAG 1
#endif

/*
Transpose data of shape [C][M][N] to [M/M_CHUNK][N/N_CHUNK][C][M_CUNK][N_CHUNK]
*/
template <typename T, unsigned M, unsigned M_CHUNK, unsigned N,
          unsigned N_CHUNK>
__global__ void
transpose(T out[M / M_CHUNK][N / N_CHUNK][COMPLEX][M_CHUNK][N_CHUNK],
          const T in[COMPLEX][M][N]) {
  const int idx_N = threadIdx.x + blockDim.x * blockIdx.x;
  const int idx_M = threadIdx.y + blockDim.y * blockIdx.y;

  static_assert(M % M_CHUNK == 0);
  static_assert(N % N_CHUNK == 0);

  if (idx_M < M && idx_N < N) {
    int m = idx_M / M_CHUNK;
    int m_c = idx_M % M_CHUNK;
    int n = idx_N / N_CHUNK;
    int n_c = idx_N % N_CHUNK;

    out[m][n][REAL][m_c][n_c] = in[REAL][idx_M][idx_N];
    out[m][n][IMAG][m_c][n_c] = in[IMAG][idx_M][idx_N];
  }
};
#endif // TRANSPOSER_KERNEL_CUH