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

#include "type_selector.h"
#include "wmma_extension.h"

// The following is a workaround for the lack of __syncwarp() in HIP<7
// The implementation is identical to that in HIP 7.0
#if defined(__HIP_PLATFORM_AMD__) && HIP_VERSION_MAJOR < 7
inline __device__ void __syncwarp() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}
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

#if defined(HAVE_ALPHA) || defined(HAVE_BETA)
  for (size_t m = 0; m < (M_PER_WARP / M_PER_WMMA); m++) {
    for (size_t n = 0; n < (N_PER_WARP / N_PER_WMMA); n++) {
      const size_t idx_m = global_idx_m(blockM, warpM, m);
      const size_t idx_n = global_idx_n(blockN, warpN, n);
#if defined(HAVE_BETA)
      wmma::fragment<wmma::accumulator, M_PER_WMMA, N_PER_WMMA, K_PER_WMMA,
                     Tshared>
          c_frag[COMPLEX];
#if defined(C_ROW_MAJOR)
      wmma::load_matrix_sync(c_frag[REAL], &(C[batch][REAL][idx_m][idx_n]),
                             N_GLOBAL, wmma::mem_row_major);
      wmma::load_matrix_sync(c_frag[IMAG], &(C[batch][IMAG][idx_m][idx_n]),
                             N_GLOBAL, wmma::mem_row_major);
#else
      wmma::load_matrix_sync(c_frag[REAL], &(C[batch][REAL][idx_n][idx_m]),
                             M_GLOBAL, wmma::mem_col_major);
      wmma::load_matrix_sync(c_frag[IMAG], &(C[batch][IMAG][idx_n][idx_m]),
                             M_GLOBAL, wmma::mem_col_major);
#endif
      __syncwarp();
#endif
      for (size_t i = 0; i < sum[REAL][m][n].num_elements; i++) {
#if defined(HAVE_ALPHA)
        const Tshared sum_real = sum[REAL][m][n].x[i];
        const Tshared sum_imag = sum[IMAG][m][n].x[i];
        sum[REAL][m][n].x[i] = static_cast<Tshared>(ALPHA_REAL) * sum_real -
                               static_cast<Tshared>(ALPHA_IMAG) * sum_imag;
        sum[IMAG][m][n].x[i] = static_cast<Tshared>(ALPHA_IMAG) * sum_real +
                               static_cast<Tshared>(ALPHA_REAL) * sum_imag;
#endif
#if defined(HAVE_BETA)
        const Tshared c_real = c_frag[REAL].x[i];
        const Tshared c_imag = c_frag[IMAG].x[i];
        sum[REAL][m][n].x[i] += static_cast<Tshared>(BETA_REAL) * c_real -
                                static_cast<Tshared>(BETA_IMAG) * c_imag;
        sum[IMAG][m][n].x[i] += static_cast<Tshared>(BETA_IMAG) * c_real +
                                static_cast<Tshared>(BETA_REAL) * c_imag;
#endif
      }
      __syncwarp();
    }
  }
#endif

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
                    const size_t &warpN, const size_t &warpK,
                    const size_t &M_TILES, const size_t &N_TILES) {

  for (size_t m = 0; m < M_TILES; m++) {
    for (size_t n = 0; n < N_TILES; n++) {
      wmma::store_matrix_sync(&C_s[warpK][REAL][warpM][warpN][0][0],
                              sum[REAL][m][n], N_PER_WMMA, wmma::mem_row_major);
      wmma::store_matrix_sync(&C_s[warpK][IMAG][warpM][warpN][0][0],
                              sum[IMAG][m][n], N_PER_WMMA, wmma::mem_row_major);

      // sum reduction over K
#if (K_SPLIT_FACTOR > 1)
      // We only need to synchronize between warps of different warpK, other
      // warpM/warpN are still independent. For now a full syncthreads is
      // easier, need to look into cooperative groups to do it properly.
      __syncthreads();
      for (size_t t = threadIdx.x; t < M_PER_WMMA * N_PER_WMMA;
           t += blockDim.x) {
        const size_t i = t / N_PER_WMMA;
        const size_t j = t % N_PER_WMMA;
        for (size_t k = 1; k < K_SPLIT_FACTOR; k++) {
          C_s[0][REAL][warpM][warpN][i][j] += C_s[k][REAL][warpM][warpN][i][j];
          C_s[0][IMAG][warpM][warpN][i][j] += C_s[k][IMAG][warpM][warpN][i][j];
        }
      }
      __syncthreads();
#else
      __syncwarp();
#endif

      size_t m_index = global_idx_m(blockM, warpM, m);
      size_t n_index = global_idx_n(blockN, warpN, n);
      for (size_t t = threadIdx.x; t < M_PER_WMMA * N_PER_WMMA;
           t += blockDim.x) {
        size_t i = t / N_PER_WMMA;
        size_t j = t % N_PER_WMMA;
        // store the submatrix, padded values are set to zero
        if (m_index + i < M_GLOBAL && n_index + j < N_GLOBAL) {
#if defined(HAVE_ALPHA)
          const Tshared sum_real = C_s[0][REAL][warpM][warpN][i][j];
          const Tshared sum_imag = C_s[0][IMAG][warpM][warpN][i][j];
          C_s[0][REAL][warpM][warpN][i][j] =
              static_cast<Tshared>(ALPHA_REAL) * sum_real -
              static_cast<Tshared>(ALPHA_IMAG) * sum_imag;
          C_s[0][IMAG][warpM][warpN][i][j] =
              static_cast<Tshared>(ALPHA_IMAG) * sum_real +
              static_cast<Tshared>(ALPHA_REAL) * sum_imag;
#endif

#if defined(C_ROW_MAJOR)
#if defined(C_COMPLEX_PLANAR)
          Tout &c_real = C[batch][REAL][m_index + i][n_index + j];
          Tout &c_imag = C[batch][IMAG][m_index + i][n_index + j];
#else
          Tout &c_real = C[batch][m_index + i][n_index + j][REAL];
          Tout &c_imag = C[batch][m_index + i][n_index + j][IMAG];
#endif
#else
#if defined(C_COMPLEX_PLANAR)
          Tout &c_real = C[batch][REAL][n_index + j][m_index + i];
          Tout &c_imag = C[batch][IMAG][n_index + j][m_index + i];
#else
          Tout &c_real = C[batch][n_index + j][m_index + i][REAL];
          Tout &c_imag = C[batch][n_index + j][m_index + i][IMAG];
#endif
#endif

#if defined(HAVE_BETA)
          const Tout c_real_copy = c_real;
          c_real = C_s[0][REAL][warpM][warpN][i][j] +
                   static_cast<Tout>(BETA_REAL) * c_real -
                   static_cast<Tout>(BETA_IMAG) * c_imag;
          c_imag = C_s[0][IMAG][warpM][warpN][i][j] +
                   static_cast<Tout>(BETA_IMAG) * c_real_copy +
                   static_cast<Tout>(BETA_REAL) * c_imag;
#else
          c_real = C_s[0][REAL][warpM][warpN][i][j];
          c_imag = C_s[0][IMAG][warpM][warpN][i][j];
#endif
        }
      }
#if K_SPLIT_FACTOR > 1
      __syncthreads();
#else
      __syncwarp();
#endif
    }
  }
}

#endif // MATRIX_OPERATIONS_H_
