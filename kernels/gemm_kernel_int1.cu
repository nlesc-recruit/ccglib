#if defined(__HIP_PLATFORM_AMD__) && (NBIT_IN == 1)
#error "1-bit GEMM is only available for NVIDIA GPUs"
#endif

#include <cuda/pipeline>
#include <cuda/std/limits>
#include <mma.h>

#include "async_copies.h"
#include "matrix_operations.h"
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

/* bit-wise negation of a tensor core fragment */
template <typename FragT> inline __device__ void negate(FragT &fragment) {
  for (auto &element : fragment.x) {
    element = ~element;
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
  const size_t warpK = threadIdx.x / WARP_SIZE;
  const size_t warpN = threadIdx.y;
  const size_t warpM = threadIdx.z;

  // number of tiles processed by each warp
  constexpr size_t M_TILES = M_PER_WARP / M_PER_WMMA;
  constexpr size_t N_TILES = N_PER_WARP / N_PER_WMMA;
  constexpr size_t K_PER_WARP = K_GLOBAL / K_SPLIT_FACTOR;
  constexpr size_t K_TILES = K_PER_WARP / K_PER_WMMA;

  wmma::fragment<wmma::accumulator, M_PER_WMMA, N_PER_WMMA, K_PER_WMMA, Tout>
      sum[COMPLEX][M_TILES][N_TILES];
  for (size_t c = 0; c < COMPLEX; c++) {
    for (size_t m = 0; m < M_TILES; m++) {
      for (size_t n = 0; n < N_TILES; n++) {
        wmma::fill_fragment(sum[c][m][n], 0);
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
        wmma::load_matrix_sync(a[c][m],
                               &(A[batch][c][global_idx_m(blockM, warpM, m)]
                                  [(warpK * K_PER_WARP + k * K_PER_WMMA) /
                                   DeviceTraits::PACKING_FACTOR]),
                               K_GLOBAL);
      }
    }

    for (size_t c = 0; c < COMPLEX; c++) {
      for (size_t n = 0; n < N_TILES; n++) {
        wmma::load_matrix_sync(b[c][n],
                               &(B[batch][c][global_idx_n(blockN, warpN, n)]
                                  [(warpK * K_PER_WARP + k * K_PER_WMMA) /
                                   DeviceTraits::PACKING_FACTOR]),
                               K_GLOBAL);
      }
    }

    // do the MMA
    // In general, MMA is D = A x B + C
    // Here, D and C are the same matrix so we have C += A x B
    // steps to do complex MMA with separate real and imaginary data
    // with x == x_r + x_i * i
    // a * b = (a_r * b_r - a_i * b_i) + (a_r * b_i + a_i * b_r) * i
    // 1. sum[real] += a_r * b_r
    // 2. sum[imag] += a_r * b_i
    // 3. b_i = - b_i (because tensor cores cannot subtract)
    // 4. sum[real] += a_i * b_i
    // 5. sum[imag] += a_i * b_r

    // step 1 and 2
    for (size_t m = 0; m < M_TILES; m++) {
      for (size_t n = 0; n < N_TILES; n++) {
#if __CUDA_ARCH__ >= 900
        bmma_sync_and(sum[REAL][m][n], a[REAL][m], b[REAL][n], sum[REAL][m][n]);
        bmma_sync_and(sum[IMAG][m][n], a[REAL][m], b[IMAG][n], sum[IMAG][m][n]);
#else
        wmma::bmma_sync(sum[REAL][m][n], a[REAL][m], b[REAL][n],
                        sum[REAL][m][n], wmma::experimental::bmmaBitOpXOR);
        wmma::bmma_sync(sum[IMAG][m][n], a[REAL][m], b[IMAG][n],
                        sum[IMAG][m][n], wmma::experimental::bmmaBitOpXOR);
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
        bmma_sync_and(sum[REAL][m][n], a[IMAG][m], b[IMAG][n], sum[REAL][m][n]);
        bmma_sync_and(sum[IMAG][m][n], a[IMAG][m], b[REAL][n], sum[IMAG][m][n]);
#else
        wmma::bmma_sync(sum[REAL][m][n], a[IMAG][m], b[IMAG][n],
                        sum[REAL][m][n], wmma::experimental::bmmaBitOpXOR);
        wmma::bmma_sync(sum[IMAG][m][n], a[IMAG][m], b[REAL][n],
                        sum[IMAG][m][n], wmma::experimental::bmmaBitOpXOR);
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
    // offset needs to be applied to IMAG part only, and only once
    size_t offset = ((c == IMAG) && (warpK == 0)) ? K_PADDING : 0;

    for (size_t m = 0; m < M_TILES; m++) {
      for (size_t n = 0; n < N_TILES; n++) {
        for (auto &element : sum[c][m][n].x) {
          // The normal XOR version of bmma_sync detects which bits in the input
          // are different, while the custom version with AND detects
          // which bits are the same. Hence, the output is flipped between these
          // two values and the final correction is different
#if __CUDA_ARCH__ >= 900
          element = 2 * (element - K_PER_WARP + offset);
#else
          element = 2 * (K_PER_WARP - offset - element);
#endif
        }
      }
    }
  }

  // store the result to global memory
#if K_SPLIT_FACTOR > 1
  __shared__ Tout C_s[K_SPLIT_FACTOR][COMPLEX][M_PER_BLOCK / M_PER_WARP]
                     [N_PER_BLOCK / N_PER_WARP][M_PER_WMMA][N_PER_WMMA];
  store_matrix_padded(sum, C, C_s, batch, blockM, warpM, blockN, warpN, warpK,
                      M_TILES, N_TILES);
#else
  store_matrix(sum, C, batch, blockM, warpM, blockN, warpN);
#endif
}

extern "C" __global__ void wmma_complex_gemm_opt(C_t C, const A_opt_t A,
                                                 const B_opt_t B) {
  const size_t batch = blockIdx.z;
  const size_t blockN = blockIdx.x;
  const size_t blockM = blockIdx.y;
  const size_t warpK = threadIdx.x / WARP_SIZE;
  const size_t warpN = threadIdx.y;
  const size_t warpM = threadIdx.z;

  constexpr size_t num_threads = block_size_x * block_size_y * block_size_z;
  const size_t tid = threadIdx.x + threadIdx.y * block_size_x +
                     threadIdx.z * block_size_x * block_size_y;

  // number of tiles processed by each warp
  constexpr size_t M_TILES = M_PER_WARP / M_PER_WMMA;
  constexpr size_t N_TILES = N_PER_WARP / N_PER_WMMA;
  constexpr size_t K_PER_WARP = K_GLOBAL / K_SPLIT_FACTOR;
  constexpr size_t K_TILES = K_PER_WARP / K_PER_WMMA;

  wmma::fragment<wmma::accumulator, M_PER_WMMA, N_PER_WMMA, K_PER_WMMA, Tout>
      sum[COMPLEX][M_TILES][N_TILES];
  for (size_t c = 0; c < COMPLEX; c++) {
    for (size_t m = 0; m < M_TILES; m++) {
      for (size_t n = 0; n < N_TILES; n++) {
        wmma::fill_fragment(sum[c][m][n], 0);
      }
    }
  }

  // shared memory buffers for partial A and B matrix. Several buffers
  // to allow for async operations: copy next submatrix to shared memory while
  // computing current submatrix
  // To save shared memory, the C matrix reuses the same shared memory in
  // case of complex-interleaved output.
  __shared__ union {
    Tin ab[A_s_size + B_s_size];
#if defined(C_COMPLEX_INTERLEAVED) || (K_SPLIT_FACTOR > 1)
    Tout c[C_s_size];
#endif
  } shmem;

  using A_s_t = Tin[NBUFFER][K_SPLIT_FACTOR][COMPLEX][M_PER_BLOCK]
                   [K_PER_WMMA / DeviceTraits::PACKING_FACTOR];
  using B_s_t = Tin[NBUFFER][K_SPLIT_FACTOR][COMPLEX][N_PER_BLOCK]
                   [K_PER_WMMA / DeviceTraits::PACKING_FACTOR];
  A_s_t &A_s = *reinterpret_cast<A_s_t *>(shmem.ab);
  B_s_t &B_s = *reinterpret_cast<B_s_t *>(&shmem.ab[A_s_size]);
#if defined(C_COMPLEX_INTERLEAVED) || (K_SPLIT_FACTOR > 1)
  using C_s_t = Tout[K_SPLIT_FACTOR][COMPLEX][M_PER_BLOCK / M_PER_WARP]
                    [N_PER_BLOCK / N_PER_WARP][M_PER_WMMA][N_PER_WMMA];
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
      copy_async<sizeof(A_s[0]), num_threads>(
          &A_s[k_buf % NBUFFER][0][0][0][0],
          &A[batch][blockM][k_buf * K_SPLIT_FACTOR][0][0], pipe, tid);
      copy_async<sizeof(B_s[0]), num_threads>(
          &B_s[k_buf % NBUFFER][0][0][0][0],
          &B[batch][blockN][k_buf * K_SPLIT_FACTOR][0][0], pipe, tid);
      pipe.producer_commit();
    }

    pipe.consumer_wait();
    __syncthreads();

    // load matrices from shared memory
    for (size_t c = 0; c < COMPLEX; c++) {
      for (size_t m = 0; m < M_TILES; m++) {
        wmma::load_matrix_sync(
            a[c][m],
            &A_s[k % NBUFFER][warpK][c][warpM * M_PER_WARP + m * M_PER_WMMA][0],
            K_PER_WMMA);
      }
    }

    for (size_t c = 0; c < COMPLEX; c++) {
      for (size_t n = 0; n < N_TILES; n++) {
        wmma::load_matrix_sync(
            b[c][n],
            &B_s[k % NBUFFER][warpK][c][warpN * N_PER_WARP + n * N_PER_WMMA][0],
            K_PER_WMMA);
      }
    }

    // do the MMA
    for (size_t m = 0; m < M_TILES; m++) {
      for (size_t n = 0; n < N_TILES; n++) {
#if __CUDA_ARCH__ >= 900
        bmma_sync_and(sum[REAL][m][n], a[REAL][m], b[REAL][n], sum[REAL][m][n]);
        bmma_sync_and(sum[IMAG][m][n], a[REAL][m], b[IMAG][n], sum[IMAG][m][n]);
#else
        wmma::bmma_sync(sum[REAL][m][n], a[REAL][m], b[REAL][n],
                        sum[REAL][m][n], wmma::experimental::bmmaBitOpXOR);
        wmma::bmma_sync(sum[IMAG][m][n], a[REAL][m], b[IMAG][n],
                        sum[IMAG][m][n], wmma::experimental::bmmaBitOpXOR);
#endif
      }
    }

    for (size_t n = 0; n < N_TILES; n++) {
      negate(b[IMAG][n]);
    }

    for (size_t m = 0; m < M_TILES; m++) {
      for (size_t n = 0; n < N_TILES; n++) {
#if __CUDA_ARCH__ >= 900
        bmma_sync_and(sum[REAL][m][n], a[IMAG][m], b[IMAG][n], sum[REAL][m][n]);
        bmma_sync_and(sum[IMAG][m][n], a[IMAG][m], b[REAL][n], sum[IMAG][m][n]);
#else
        wmma::bmma_sync(sum[REAL][m][n], a[IMAG][m], b[IMAG][n],
                        sum[REAL][m][n], wmma::experimental::bmmaBitOpXOR);
        wmma::bmma_sync(sum[IMAG][m][n], a[IMAG][m], b[REAL][n],
                        sum[IMAG][m][n], wmma::experimental::bmmaBitOpXOR);
#endif
      }
    }

    pipe.consumer_release();

    __syncthreads();
  }

  // fix output values
  for (size_t c = 0; c < COMPLEX; c++) {
    // offset needs to be applied to IMAG part only, and only once
    size_t offset = ((c == IMAG) && (warpK == 0)) ? K_PADDING : 0;

    for (size_t m = 0; m < M_TILES; m++) {
      for (size_t n = 0; n < N_TILES; n++) {
        for (auto &element : sum[c][m][n].x) {
#if __CUDA_ARCH__ >= 900
          element = 2 * (element - K_PER_WARP + offset);
#else
          element = 2 * (K_PER_WARP - offset - element);
#endif
        }
      }
    }
  }

  // store the result to global memory
#if defined(C_COMPLEX_INTERLEAVED) || (K_SPLIT_FACTOR > 1)
  store_matrix_padded(sum, C, C_s, batch, blockM, warpM, blockN, warpN, warpK,
                      M_TILES, N_TILES);
#else
  store_matrix(sum, C, batch, blockM, warpM, blockN, warpN);
#endif
}
