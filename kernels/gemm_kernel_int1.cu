#if defined(__HIP_PLATFORM_AMD__) && (NBIT_IN == 1)
#error "1-bit GEMM is only available for NVIDIA GPUs"
#endif

#include <cuda/pipeline>
#include <cuda/std/limits>
#include <mma.h>

#include "async_copies.h"
#include "wmma_extension.h"

#include "type_selector.h"

#if NBIT_IN != 1
#error NBIT_IN must be 1
#endif

// Memory layout of A must be row-major, B col-major
#ifdef A_COL_MAJOR
#error "A matrix must be row-major for 1-bit GEMM"
#endif
#ifdef B_ROW_MAJOR
#error "B matrix must be col-major for 1-bit GEMM"
#endif

// data layout for optimal transfer to shared memory
using A_opt_t =
    Tin[BATCH_SIZE][M_GLOBAL / M_PER_BLOCK][K_GLOBAL / K_PER_WMMA][COMPLEX]
       [M_PER_BLOCK][K_PER_WMMA / DeviceTraits::PACKING_FACTOR];
using B_opt_t =
    Tin[BATCH_SIZE][N_GLOBAL / N_PER_BLOCK][K_GLOBAL / K_PER_WMMA][COMPLEX]
       [N_PER_BLOCK][K_PER_WMMA / DeviceTraits::PACKING_FACTOR];

#if defined(C_COMPLEX_MIDDLE)
#ifdef C_ROW_MAJOR
using C_t = Tout[BATCH_SIZE][COMPLEX][M_GLOBAL][N_GLOBAL];
#else
using C_t = Tout[BATCH_SIZE][COMPLEX][N_GLOBAL][M_GLOBAL];
#endif
#elif defined(C_COMPLEX_LAST)
#ifdef C_ROW_MAJOR
using C_t = Tout[BATCH_SIZE][M_GLOBAL][N_GLOBAL][COMPLEX];
#else
using C_t = Tout[BATCH_SIZE][N_GLOBAL][M_GLOBAL][COMPLEX];
#endif
#endif

inline __device__ size_t global_idx_m(const size_t &blockM, const size_t &warpM,
                                      const size_t &m) {
  return blockM * M_PER_BLOCK + warpM * M_PER_WARP + m * M_PER_WMMA;
}

inline __device__ size_t global_idx_n(const size_t &blockN, const size_t &warpN,
                                      const size_t &n) {
  return blockN * N_PER_BLOCK + warpN * N_PER_WARP + n * N_PER_WMMA;
}

/* bit-wise negation of a tensor core fragment */
template <typename FragT> inline __device__ void negate(FragT &fragment) {
  for (auto &element : fragment.x) {
    element = ~element;
  }
}

inline __device__ void store_matrix(
    wmma::fragment<wmma::accumulator, M_PER_WMMA, N_PER_WMMA, K_PER_WMMA, Tout>
        accumulator[COMPLEX][M_PER_WARP / M_PER_WMMA][N_PER_WARP / N_PER_WMMA],
    C_t C, const size_t &batch, const size_t &blockM, const size_t &warpM,
    const size_t &blockN, const size_t &warpN) {
  for (size_t c = 0; c < COMPLEX; c++) {
    for (size_t m = 0; m < (M_PER_WARP / M_PER_WMMA); m++) {
      for (size_t n = 0; n < (N_PER_WARP / N_PER_WMMA); n++) {
        const size_t idx_m = global_idx_m(blockM, warpM, m);
        const size_t idx_n = global_idx_n(blockN, warpN, n);
#ifdef C_ROW_MAJOR
        wmma::store_matrix_sync(&(C[batch][c][idx_m][idx_n]),
                                accumulator[c][m][n], N_GLOBAL,
                                wmma::mem_row_major);
#else
        wmma::store_matrix_sync(&(C[batch][c][idx_n][idx_m]),
                                accumulator[c][m][n], M_GLOBAL,
                                wmma::mem_col_major);
#endif
      }
    }
  }
}

// Starting with the hopper generation, bmma with XOR is no longer available in
// hardware With this custom implementation, the result of bmma_sync is slightly
// different than the XOR version, but this is easily corrected in the final
// output.
#if __CUDA_ARCH__ >= 900
static inline __device__ void bmma_sync_and(
    wmma::fragment<wmma::accumulator, M_PER_WMMA, N_PER_WMMA, K_PER_WMMA, int>
        &d,
    wmma::fragment<wmma::matrix_a, M_PER_WMMA, N_PER_WMMA, K_PER_WMMA,
                   wmma::experimental::precision::b1, wmma::row_major> &a,
    wmma::fragment<wmma::matrix_b, M_PER_WMMA, N_PER_WMMA, K_PER_WMMA,
                   wmma::experimental::precision::b1, wmma::col_major> &b,
    const wmma::fragment<wmma::accumulator, M_PER_WMMA, N_PER_WMMA, K_PER_WMMA,
                         int> &c) {
  // AND, flip inputs a and b, AND again, flip inputs back
  wmma::bmma_sync(d, a, b, c, wmma::experimental::bmmaBitOpAND);
  negate(a);
  negate(b);
  wmma::bmma_sync(d, a, b, c, wmma::experimental::bmmaBitOpAND);
  negate(a);
  negate(b);
}
#endif

extern "C" __global__ void wmma_complex_gemm_basic(C_t &C, const A_t &A,
                                                   const B_t &B) {
  const size_t batch = blockIdx.z;
  const size_t blockN = blockIdx.x;
  const size_t blockM = blockIdx.y;
  const size_t warpN = threadIdx.y;
  const size_t warpM = threadIdx.z;

  // number of tiles processed by each warp
  constexpr size_t M_TILES = M_PER_WARP / M_PER_WMMA;
  constexpr size_t N_TILES = N_PER_WARP / N_PER_WMMA;
  constexpr size_t K_TILES = K_GLOBAL / K_PER_WMMA;

  wmma::fragment<wmma::accumulator, M_PER_WMMA, N_PER_WMMA, K_PER_WMMA, Tout>
      accumulator[COMPLEX][M_TILES][N_TILES];
  for (size_t c = 0; c < COMPLEX; c++) {
    for (size_t m = 0; m < M_TILES; m++) {
      for (size_t n = 0; n < N_TILES; n++) {
        wmma::fill_fragment(accumulator[c][m][n], 0);
      }
    }
  }

  for (size_t k = 0; k < K_TILES; k++) {
    // declare input fragments
    wmma::fragment<wmma::matrix_a, M_PER_WMMA, N_PER_WMMA, K_PER_WMMA, Ttc,
                   wmma::row_major>
        a[COMPLEX][M_TILES];
    wmma::fragment<wmma::matrix_b, M_PER_WMMA, N_PER_WMMA, K_PER_WMMA, Ttc,
                   wmma::col_major>
        b[COMPLEX][N_TILES];

    // load matrices from global memory
    for (size_t c = 0; c < COMPLEX; c++) {
      for (size_t m = 0; m < M_TILES; m++) {
        wmma::load_matrix_sync(
            a[c][m],
            &(A[batch][c][global_idx_m(blockM, warpM, m)]
               [k * K_PER_WMMA / DeviceTraits::PACKING_FACTOR]),
            K_GLOBAL);
      }
    }

    for (size_t c = 0; c < COMPLEX; c++) {
      for (size_t n = 0; n < N_TILES; n++) {
        wmma::load_matrix_sync(
            b[c][n],
            &(B[batch][c][global_idx_n(blockN, warpN, n)]
               [k * K_PER_WMMA / DeviceTraits::PACKING_FACTOR]),
            K_GLOBAL);
      }
    }

    // do the MMA
    // In general, MMA is D = A x B + C
    // Here, D and C are the same matrix so we have C += A x B
    // steps to do complex MMA with separate real and imaginary data
    // with x == x_r + x_i * i
    // a * b = (a_r * b_r - a_i * b_i) + (a_r * b_i + a_i * b_r) * i
    // 1. accumulator[real] += a_r * b_r
    // 2. accumulator[imag] += a_r * b_i
    // 3. b_i = - b_i (because tensor cores cannot subtract)
    // 4. accumulator[real] += a_i * b_i
    // 5. accumulator[imag] += a_i * b_r

    // step 1 and 2
    for (size_t m = 0; m < M_TILES; m++) {
      for (size_t n = 0; n < N_TILES; n++) {
#if __CUDA_ARCH__ >= 900
        bmma_sync_and(accumulator[REAL][m][n], a[REAL][m], b[REAL][n],
                      accumulator[REAL][m][n]);
        bmma_sync_and(accumulator[IMAG][m][n], a[REAL][m], b[IMAG][n],
                      accumulator[IMAG][m][n]);
#else
        wmma::bmma_sync(accumulator[REAL][m][n], a[REAL][m], b[REAL][n],
                        accumulator[REAL][m][n],
                        wmma::experimental::bmmaBitOpXOR);
        wmma::bmma_sync(accumulator[IMAG][m][n], a[REAL][m], b[IMAG][n],
                        accumulator[IMAG][m][n],
                        wmma::experimental::bmmaBitOpXOR);
#endif
      }
    }

    // step 3
    for (size_t n = 0; n < N_TILES; n++) {
      negate(b[IMAG][n]);
    }

    // step 4 and 5
    for (size_t m = 0; m < M_TILES; m++) {
      for (size_t n = 0; n < N_TILES; n++) {
#if __CUDA_ARCH__ >= 900
        bmma_sync_and(accumulator[REAL][m][n], a[IMAG][m], b[IMAG][n],
                      accumulator[REAL][m][n]);
        bmma_sync_and(accumulator[IMAG][m][n], a[IMAG][m], b[REAL][n],
                      accumulator[IMAG][m][n]);
#else
        wmma::bmma_sync(accumulator[REAL][m][n], a[IMAG][m], b[IMAG][n],
                        accumulator[REAL][m][n],
                        wmma::experimental::bmmaBitOpXOR);
        wmma::bmma_sync(accumulator[IMAG][m][n], a[IMAG][m], b[REAL][n],
                        accumulator[IMAG][m][n],
                        wmma::experimental::bmmaBitOpXOR);
#endif
      }
    }
  }

  // Fix result: a dot b = K - 2 * popc(a xor b)
  // 2 K here because we do two TC operations per accumulator fragment, so 2 K
  // values were added together should also take care of padding: extra samples
  // are zero, interpreted as -1. For the REAL part of the output: first -1 * -1
  // is added, then -1 * 1 due to negation of bi. Effectively no correction
  // needed for the IMAG part, -1 * -1 is added twice, so we need to subtract
  // the amount of padding twice hence we end up with a dot b (real part) = 2 *
  // (K - popc(a xor b)) a dot b (imag part) = 2 * (K - K_PADDING - popc(a xor
  // b))
  for (size_t c = 0; c < COMPLEX; c++) {
    size_t offset = c == REAL ? 0 : K_PADDING;

    for (size_t m = 0; m < M_TILES; m++) {
      for (size_t n = 0; n < N_TILES; n++) {
        for (auto &element : accumulator[c][m][n].x) {
          // The normal XOR version of bmma_sync detects which bits in the input
          // are different, while the custom version with AND detects
          // which bits are the same. Hence, the output is flipped between these
          // two values and the final correction is different
#if __CUDA_ARCH__ >= 900
          element = 2 * (element - K_GLOBAL + offset);
#else
          element = 2 * (K_GLOBAL - offset - element);
#endif
        }
      }
    }
  }

  // store the result to global memory
  store_matrix(accumulator, C, batch, blockM, warpM, blockN, warpN);
}

extern "C" __global__ void wmma_complex_gemm_opt(C_t C, const A_opt_t A,
                                                 const B_opt_t B) {
  const size_t batch = blockIdx.z;
  const size_t blockN = blockIdx.x;
  const size_t blockM = blockIdx.y;
  const size_t warpN = threadIdx.y;
  const size_t warpM = threadIdx.z;

  constexpr size_t num_threads = block_size_x * block_size_y * block_size_z;
  const size_t tid = threadIdx.x + threadIdx.y * block_size_x +
                     threadIdx.z * block_size_x * block_size_y;

  // number of tiles processed by each warp
  constexpr size_t M_TILES = M_PER_WARP / M_PER_WMMA;
  constexpr size_t N_TILES = N_PER_WARP / N_PER_WMMA;
  constexpr size_t K_TILES = K_GLOBAL / K_PER_WMMA;

  wmma::fragment<wmma::accumulator, M_PER_WMMA, N_PER_WMMA, K_PER_WMMA, Tout>
      accumulator[COMPLEX][M_TILES][N_TILES];
  for (size_t c = 0; c < COMPLEX; c++) {
    for (size_t m = 0; m < M_TILES; m++) {
      for (size_t n = 0; n < N_TILES; n++) {
        wmma::fill_fragment(accumulator[c][m][n], 0);
      }
    }
  }

  // shared memory buffers for partial A and B matrix. Several buffers
  // to allow for async operations: copy next submatrix to shared memory while
  // computing current submatrix
  // To save shared memory, the C matrix reuses the same shared memory in
  // case of complex-last output.
  constexpr size_t A_s_size = NBUFFER * COMPLEX * M_PER_BLOCK * K_PER_WMMA /
                              DeviceTraits::PACKING_FACTOR;
  constexpr size_t B_s_size = NBUFFER * COMPLEX * N_PER_BLOCK * K_PER_WMMA /
                              DeviceTraits::PACKING_FACTOR;
#if defined(C_COMPLEX_LAST)
  constexpr size_t C_s_size = (M_PER_BLOCK / M_PER_WARP) *
                              (N_PER_BLOCK / N_PER_WARP) * M_PER_WMMA *
                              N_PER_WMMA;
#endif
  __shared__ union {
    Tin ab[A_s_size + B_s_size];
#if defined(C_COMPLEX_LAST)
    Tout c[C_s_size];
#endif
  } shmem;

  using A_s_t = Tin[NBUFFER][COMPLEX][M_PER_BLOCK]
                   [K_PER_WMMA / DeviceTraits::PACKING_FACTOR];
  using B_s_t = Tin[NBUFFER][COMPLEX][N_PER_BLOCK]
                   [K_PER_WMMA / DeviceTraits::PACKING_FACTOR];
  A_s_t &A_s = *reinterpret_cast<A_s_t *>(shmem.ab);
  B_s_t &B_s = *reinterpret_cast<B_s_t *>(&shmem.ab[A_s_size]);
#if defined(C_COMPLEX_LAST)
  using C_s_t = Tout[M_PER_BLOCK / M_PER_WARP][N_PER_BLOCK / N_PER_WARP]
                    [M_PER_WMMA][N_PER_WMMA];
  C_s_t &C_s = *reinterpret_cast<C_s_t *>(shmem.c);
#endif

  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

  for (size_t k = 0, k_buf = 0; k < K_TILES; k++) {
    // declare input fragments
    wmma::fragment<wmma::matrix_a, M_PER_WMMA, N_PER_WMMA, K_PER_WMMA, Ttc,
                   wmma::row_major>
        a[COMPLEX][M_TILES];
    wmma::fragment<wmma::matrix_b, M_PER_WMMA, N_PER_WMMA, K_PER_WMMA, Ttc,
                   wmma::col_major>
        b[COMPLEX][N_TILES];

    // copy next data to smem
    for (; k_buf < K_TILES && k_buf < (k + NBUFFER); k_buf++) {
      pipe.producer_acquire();
      copy_async<sizeof(A_s[0]), num_threads>(&A_s[k_buf % NBUFFER][0][0][0],
                                              &A[batch][blockM][k_buf][0][0],
                                              pipe, tid);
      copy_async<sizeof(B_s[0]), num_threads>(&B_s[k_buf % NBUFFER][0][0][0],
                                              &B[batch][blockN][k_buf][0][0],
                                              pipe, tid);
      pipe.producer_commit();
    }

    pipe.consumer_wait();
    __syncthreads();

    // load matrices from shared memory
    for (size_t c = 0; c < COMPLEX; c++) {
      for (size_t m = 0; m < M_TILES; m++) {
        wmma::load_matrix_sync(
            a[c][m],
            &A_s[k % NBUFFER][c][warpM * M_PER_WARP + m * M_PER_WMMA][0],
            K_PER_WMMA);
      }
    }

    for (size_t c = 0; c < COMPLEX; c++) {
      for (size_t n = 0; n < N_TILES; n++) {
        wmma::load_matrix_sync(
            b[c][n],
            &B_s[k % NBUFFER][c][warpN * N_PER_WARP + n * N_PER_WMMA][0],
            K_PER_WMMA);
      }
    }

    // do the MMA
    for (size_t m = 0; m < M_TILES; m++) {
      for (size_t n = 0; n < N_TILES; n++) {
#if __CUDA_ARCH__ >= 900
        bmma_sync_and(accumulator[REAL][m][n], a[REAL][m], b[REAL][n],
                      accumulator[REAL][m][n]);
        bmma_sync_and(accumulator[IMAG][m][n], a[REAL][m], b[IMAG][n],
                      accumulator[IMAG][m][n]);
#else
        wmma::bmma_sync(accumulator[REAL][m][n], a[REAL][m], b[REAL][n],
                        accumulator[REAL][m][n],
                        wmma::experimental::bmmaBitOpXOR);
        wmma::bmma_sync(accumulator[IMAG][m][n], a[REAL][m], b[IMAG][n],
                        accumulator[IMAG][m][n],
                        wmma::experimental::bmmaBitOpXOR);
#endif
      }
    }

    for (size_t n = 0; n < N_TILES; n++) {
      negate(b[IMAG][n]);
    }

    for (size_t m = 0; m < M_TILES; m++) {
      for (size_t n = 0; n < N_TILES; n++) {
#if __CUDA_ARCH__ >= 900
        bmma_sync_and(accumulator[REAL][m][n], a[IMAG][m], b[IMAG][n],
                      accumulator[REAL][m][n]);
        bmma_sync_and(accumulator[IMAG][m][n], a[IMAG][m], b[REAL][n],
                      accumulator[IMAG][m][n]);
#else
        wmma::bmma_sync(accumulator[REAL][m][n], a[IMAG][m], b[IMAG][n],
                        accumulator[REAL][m][n],
                        wmma::experimental::bmmaBitOpXOR);
        wmma::bmma_sync(accumulator[IMAG][m][n], a[IMAG][m], b[REAL][n],
                        accumulator[IMAG][m][n],
                        wmma::experimental::bmmaBitOpXOR);
#endif
      }
    }

    pipe.consumer_release();

    __syncthreads();
  }

  // fix output values
  for (size_t c = 0; c < COMPLEX; c++) {
    size_t offset = c == REAL ? 0 : K_PADDING;

    for (size_t m = 0; m < M_TILES; m++) {
      for (size_t n = 0; n < N_TILES; n++) {
        for (auto &element : accumulator[c][m][n].x) {
#if __CUDA_ARCH__ >= 900
          element = 2 * (element - K_GLOBAL + offset);
#else
          element = 2 * (K_GLOBAL - offset - element);
#endif
        }
      }
    }
  }

  // store the result to global memory
#if defined(C_COMPLEX_MIDDLE)
  store_matrix(accumulator, C, batch, blockM, warpM, blockN, warpN);
#elif defined(C_COMPLEX_LAST)
  for (size_t m = 0; m < M_TILES; m++) {
    for (size_t n = 0; n < N_TILES; n++) {
      for (size_t c = 0; c < COMPLEX; c++) {
        wmma::store_matrix_sync(&C_s[warpM][warpN][0][0], accumulator[c][m][n],
                                N_PER_WMMA, wmma::mem_row_major);
        __syncwarp();
        size_t m_index = global_idx_m(blockM, warpM, m);
        size_t n_index = global_idx_n(blockN, warpN, n);
        for (size_t t = threadIdx.x; t < M_PER_WMMA * N_PER_WMMA;
             t += WARP_SIZE) {
          size_t i = t / N_PER_WMMA;
          size_t j = t % N_PER_WMMA;
          // store the submatrix
          if (m_index + i < M_GLOBAL && n_index + j < N_GLOBAL) {
#ifdef C_ROW_MAJOR
            C[batch][m_index + i][n_index + j][c] = C_s[warpM][warpN][i][j];
#else
            C[batch][n_index + j][m_index + i][c] = C_s[warpM][warpN][i][j];
#endif
          }
        }
        __syncwarp();
      }
    }
  }
#endif
}
