#include <cuda/pipeline>
#include <cuda/std/limits>
#include <mma.h>

#include "async_copies.h"
#include "wmma_extension.h"

using namespace nvcuda;

#ifndef COMPLEX
#define COMPLEX 2
#endif
#ifndef REAL
#define REAL 0
#endif
#ifndef IMAG
#define IMAG 1
#endif

// All values related to data layout must be defined at compile time
#if !defined BATCH_SIZE || !defined M_GLOBAL || !defined N_GLOBAL ||           \
    !defined K_GLOBAL
#error                                                                         \
    "BATCH_SIZE, M_GLOBAL, N_GLOBAL, and K_GLOBAL values per block, warp, tensor core must be defined at compile time"
#endif

#if NBIT == 1
using Tin = unsigned int;
using Ttc = wmma::experimental::precision::b1;
using Tout = int;
#else
#error NBIT must be 1
#endif

// Number of samples in one element of type Tin
#define PACKING_FACTOR (sizeof(Tin) * CHAR_BIT / NBIT)

// basic data layout
using A_t = Tin[BATCH_SIZE][COMPLEX][M_GLOBAL][K_GLOBAL / PACKING_FACTOR];
using B_t = Tin[BATCH_SIZE][COMPLEX][N_GLOBAL][K_GLOBAL / PACKING_FACTOR];
// data layout for optimal transfer to shared memory
using A_opt_t = Tin[BATCH_SIZE][M_GLOBAL / M_PER_BLOCK][K_GLOBAL / K_PER_WMMA]
                   [COMPLEX][M_PER_BLOCK][K_PER_WMMA / PACKING_FACTOR];
using B_opt_t = Tin[BATCH_SIZE][N_GLOBAL / N_PER_BLOCK][K_GLOBAL / K_PER_WMMA]
                   [COMPLEX][N_PER_BLOCK][K_PER_WMMA / PACKING_FACTOR];

using C_t = Tout[BATCH_SIZE][COMPLEX][M_GLOBAL][N_GLOBAL];

// Starting with the hopper generation, bmma with XOR is no longer available in
// hardware With this custom implementation, the result of bmma_sync is slightly
// different than the XOR version, but this is easily corrected in the final
// output. After this function is run, the data in the a and b fragments is
// flipped
#if __CUDA_ARCH__ >= 900
static inline __device__ void bmma_sync_and(
    wmma::fragment<wmma::accumulator, 16, 8, 256, int> &d,
    wmma::fragment<wmma::matrix_a, 16, 8, 256,
                   wmma::experimental::precision::b1, wmma::row_major> &a,
    wmma::fragment<wmma::matrix_b, 16, 8, 256,
                   wmma::experimental::precision::b1, wmma::col_major> &b,
    const wmma::fragment<wmma::accumulator, 16, 8, 256, int> &c) {
  // AND, flip inputs a and b, AND again
  wmma::bmma_sync(d, a, b, c, wmma::experimental::bmmaBitOpAND);
  __syncwarp();
  for (auto &element : a.x) {
    element = ~element;
  }
  for (auto &element : b.x) {
    element = ~element;
  }
  __syncwarp();
  wmma::bmma_sync(d, a, b, c, wmma::experimental::bmmaBitOpAND);
}
#endif

extern "C" __global__ void wmma_complex_gemm_basic(C_t &C, const A_t &A,
                                                   const B_t &B) {
  const unsigned batch = blockIdx.z;
  const size_t blockN = blockIdx.x;
  const size_t blockM = blockIdx.y;
  const size_t warpN = threadIdx.y;
  const size_t warpM = threadIdx.z;

  // number of tiles processed by each warp
  constexpr size_t M_TILES = M_PER_WARP / M_PER_WMMA;
  constexpr size_t N_TILES = N_PER_WARP / N_PER_WMMA;
  constexpr size_t K_TILES = K_GLOBAL / K_PER_WMMA;

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
        wmma::load_matrix_sync(
            a[c][m],
            &(A[batch][c][blockM * M_PER_BLOCK + warpM * M_PER_WARP +
                          m * M_PER_WMMA][k * K_PER_WMMA / PACKING_FACTOR]),
            K_GLOBAL);
      }
    }

    for (size_t c = 0; c < COMPLEX; c++) {
      for (size_t n = 0; n < N_TILES; n++) {
        wmma::load_matrix_sync(
            b[c][n],
            &(B[batch][c][blockN * N_PER_BLOCK + warpN * N_PER_WARP +
                          n * N_PER_WMMA][k * K_PER_WMMA / PACKING_FACTOR]),
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
    __syncwarp();
    for (size_t n = 0; n < N_TILES; n++) {
      for (auto &element : b[IMAG][n].x) {
        element = ~element;
      }
    }
    __syncwarp();

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
  // 2 K here because we do two TC operations per sum fragment, so 2 K values
  // were added together should also take care of padding: extra samples are
  // zero, interpreted as -1. For the REAL part of the output: first -1 * -1 is
  // added, then -1 * 1 due to negation of bi. Effectively no correction needed
  // for the IMAG part, -1 * -1 is added twice, so we need to subtract the
  // amount of padding twice hence we end up with
  // a dot b (real part) = 2 * (K - popc(a xor b))
  // a dot b (imag part) = 2 * (K - K_PADDING - popc(a xor b))
  __syncwarp();
  for (size_t c = 0; c < COMPLEX; c++) {
    unsigned offset = c == REAL ? 0 : K_PADDING;

    for (size_t m = 0; m < M_TILES; m++) {
      for (size_t n = 0; n < N_TILES; n++) {
        for (auto &element : sum[c][m][n].x) {
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
  __syncwarp();

  // store the result to global memory
  for (size_t c = 0; c < COMPLEX; c++) {
    for (size_t m = 0; m < M_TILES; m++) {
      for (size_t n = 0; n < N_TILES; n++) {
        Tout *c_ptr =
            &(C[batch][c]
               [blockM * M_PER_BLOCK + warpM * M_PER_WARP + m * M_PER_WMMA]
               [blockN * N_PER_BLOCK + warpN * N_PER_WARP + n * N_PER_WMMA]);
        wmma::store_matrix_sync(c_ptr, sum[c][m][n], N_GLOBAL,
                                wmma::mem_row_major);
      }
    }
  }
}

extern "C" __global__ void wmma_complex_gemm_opt(C_t C, const A_opt_t A,
                                                 const B_opt_t B) {
  const unsigned batch = blockIdx.z;
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
      sum[COMPLEX][M_TILES][N_TILES];
  for (size_t c = 0; c < COMPLEX; c++) {
    for (size_t m = 0; m < M_TILES; m++) {
      for (size_t n = 0; n < N_TILES; n++) {
        wmma::fill_fragment(sum[c][m][n], 0);
      }
    }
  }

  __shared__ Tin
      A_s[NBUFFER][COMPLEX][M_PER_BLOCK][K_PER_WMMA / PACKING_FACTOR];
  __shared__ Tin
      B_s[NBUFFER][COMPLEX][N_PER_BLOCK][K_PER_WMMA / PACKING_FACTOR];

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

    __syncwarp();
    for (size_t n = 0; n < N_TILES; n++) {
      for (auto &element : b[IMAG][n].x) {
        element = ~element;
      }
    }
    __syncwarp();

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
  __syncwarp();
  for (size_t c = 0; c < COMPLEX; c++) {
    unsigned offset = c == REAL ? 0 : K_PADDING;

    for (size_t m = 0; m < M_TILES; m++) {
      for (size_t n = 0; n < N_TILES; n++) {
        for (auto &element : sum[c][m][n].x) {
#if __CUDA_ARCH__ >= 900
          element = 2 * (element - K_GLOBAL + offset);
#else
          element = 2 * (K_GLOBAL - offset - element);
#endif
        }
      }
    }
  }
  __syncwarp();

  // store the result to global memory
  for (size_t c = 0; c < COMPLEX; c++) {
    for (size_t m = 0; m < M_TILES; m++) {
      for (size_t n = 0; n < N_TILES; n++) {
        Tout *c_ptr =
            &C[batch][c]
              [blockM * M_PER_BLOCK + warpM * M_PER_WARP + m * M_PER_WMMA]
              [blockN * N_PER_BLOCK + warpN * N_PER_WARP + n * N_PER_WMMA];
        wmma::store_matrix_sync(c_ptr, sum[c][m][n], N_GLOBAL,
                                wmma::mem_row_major);
      }
    }
  }
}
