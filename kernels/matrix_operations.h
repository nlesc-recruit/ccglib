#ifndef MATRIX_OPERATIONS_H_
#define MATRIX_OPERATIONS_H_

#if defined(__HIP_PLATFORM_AMD__)
#include <limits.h>
#include <rocwmma/rocwmma.hpp>
namespace wmma = rocwmma;
#else
#include <cuda/pipeline>
#include <mma.h>
using namespace nvcuda;
#endif

// The following is a workaround for the lack of __syncwarp() in HIP
#if defined(__HIP_PLATFORM_AMD__)
inline __device__ void __syncwarp(){};
#endif

inline __device__ size_t global_idx_m(const size_t &blockM, const size_t &warpM,
                                      const size_t &m) {
  return blockM * M_PER_BLOCK + warpM * M_PER_WARP + m * M_PER_WMMA;
}

inline __device__ size_t global_idx_n(const size_t &blockN, const size_t &warpN,
                                      const size_t &n) {
  return blockN * N_PER_BLOCK + warpN * N_PER_WARP + n * N_PER_WMMA;
}

#if !(REQUIRES_DOWNCAST)
inline __device__ void
store_matrix(Accumulator_t sum[COMPLEX][M_PER_WARP / M_PER_WMMA]
                              [N_PER_WARP / N_PER_WMMA],
             C_t C, const size_t &batch, const size_t &blockM,
             const size_t &warpM, const size_t &blockN, const size_t &warpN) {

  for (size_t c = 0; c < COMPLEX; c++) {
    for (size_t m = 0; m < (M_PER_WARP / M_PER_WMMA); m++) {
      for (size_t n = 0; n < (N_PER_WARP / N_PER_WMMA); n++) {
        const size_t idx_m = global_idx_m(blockM, warpM, m);
        const size_t idx_n = global_idx_n(blockN, warpN, n);
#if defined(C_ROW_MAJOR)
        wmma::store_matrix_sync(&(C[batch][c][idx_m][idx_n]), sum[c][m][n],
                                N_GLOBAL, wmma::mem_row_major);
#else
        wmma::store_matrix_sync(&(C[batch][c][idx_n][idx_m]), sum[c][m][n],
                                M_GLOBAL, wmma::mem_col_major);
#endif
      }
    }
  }
}
#endif

template <typename T>
inline __device__ void
store_matrix_padded(Accumulator_t sum[COMPLEX][M_PER_WARP / M_PER_WMMA]
                                     [N_PER_WARP / N_PER_WMMA],
                    C_t C, T &C_s, const size_t &batch, const size_t &blockM,
                    const size_t &warpM, const size_t &blockN,
                    const size_t &warpN, const size_t &M_TILES,
                    const size_t &N_TILES) {

#if defined(C_COMPLEX_MIDDLE)
  for (size_t c = 0; c < COMPLEX; c++) {
    for (size_t m = 0; m < M_TILES; m++) {
      for (size_t n = 0; n < N_TILES; n++) {
#else
  for (size_t m = 0; m < M_TILES; m++) {
    for (size_t n = 0; n < N_TILES; n++) {
      for (size_t c = 0; c < COMPLEX; c++) {
#endif
        wmma::store_matrix_sync(&C_s[warpM][warpN][0][0], sum[c][m][n],
                                N_PER_WMMA, wmma::mem_row_major);

        __syncwarp();
        size_t m_index = global_idx_m(blockM, warpM, m);
        size_t n_index = global_idx_n(blockN, warpN, n);
        for (size_t t = threadIdx.x; t < M_PER_WMMA * N_PER_WMMA;
             t += WARP_SIZE) {
          size_t i = t / N_PER_WMMA;
          size_t j = t % N_PER_WMMA;
          // store the submatrix, padded values are set to zero
          if (m_index + i < M_GLOBAL && n_index + j < N_GLOBAL) {
#if defined(C_ROW_MAJOR)
#if defined(C_COMPLEX_MIDDLE)
            C[batch][c][m_index + i][n_index + j] = C_s[warpM][warpN][i][j];
#else
            C[batch][m_index + i][n_index + j][c] = C_s[warpM][warpN][i][j];
#endif
#else
#if defined(C_COMPLEX_MIDDLE)
            C[batch][c][n_index + j][m_index + i] = C_s[warpM][warpN][i][j];
#else
            C[batch][n_index + j][m_index + i][c] = C_s[warpM][warpN][i][j];
#endif
#endif
          }
        }
        __syncwarp();
      }
    }
  }
}

#endif // MATRIX_OPERATIONS_H_
