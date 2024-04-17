#include <cuda/pipeline>
#include <cuda_fp16.h>
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
#if !defined M_GLOBAL || !defined N_GLOBAL || !defined K_GLOBAL
#error                                                                         \
    "M_GLOBAL, N_GLOBAL, K_GLOBAL and values per block, warp, tensor core must be defined at compile time"
#endif

#if NBIT == 16
using Tin = half;
using Ttc = half;
using Tout = float;
#else
#error NBIT must be 16
#endif

// basic data layout
using A_t = Tin[COMPLEX][M_GLOBAL][K_GLOBAL];
using B_t = Tin[COMPLEX][N_GLOBAL][K_GLOBAL];
// data layout for optimal transfer to shared memory
using A_opt_t = Tin[M_GLOBAL / M_PER_BLOCK][K_GLOBAL / K_PER_WMMA][COMPLEX]
                   [M_PER_BLOCK][K_PER_WMMA];
using B_opt_t = Tin[N_GLOBAL / N_PER_BLOCK][K_GLOBAL / K_PER_WMMA][COMPLEX]
                   [N_PER_BLOCK][K_PER_WMMA];
using C_t = Tout[COMPLEX][M_GLOBAL][N_GLOBAL];

extern "C" __global__ void wmma_complex_gemm_basic(C_t C, const A_t A,
                                                   const B_t B) {
  const unsigned blockN = blockIdx.x;
  const unsigned blockM = blockIdx.y;
  const unsigned warpN = threadIdx.y;
  const unsigned warpM = threadIdx.z;

  // number of tiles processed by each warp
  constexpr unsigned M_TILES = M_PER_WARP / M_PER_WMMA;
  constexpr unsigned N_TILES = N_PER_WARP / N_PER_WMMA;
  constexpr unsigned K_TILES = K_GLOBAL / K_PER_WMMA;

  wmma::fragment<wmma::accumulator, M_PER_WMMA, N_PER_WMMA, K_PER_WMMA, Tout>
      sum[COMPLEX][M_TILES][N_TILES];
  for (int c = 0; c < COMPLEX; c++) {
    for (int m = 0; m < M_TILES; m++) {
      for (int n = 0; n < N_TILES; n++) {
        wmma::fill_fragment(sum[c][m][n], 0);
      }
    }
  }

  for (int k = 0; k < K_TILES; k++) {
    // declare input fragments
    wmma::fragment<wmma::matrix_a, M_PER_WMMA, N_PER_WMMA, K_PER_WMMA, Ttc,
                   wmma::row_major>
        a[COMPLEX][M_TILES];
    wmma::fragment<wmma::matrix_b, M_PER_WMMA, N_PER_WMMA, K_PER_WMMA, Ttc,
                   wmma::col_major>
        b[COMPLEX][N_TILES];

    // load matrices from global memory
    for (int c = 0; c < COMPLEX; c++) {
      for (int m = 0; m < M_TILES; m++) {
        int k_index = k * K_PER_WMMA;
        wmma::load_matrix_sync(a[c][m],
                               &A[c][blockM * M_PER_BLOCK + warpM * M_PER_WARP +
                                     m * M_PER_WMMA][k_index],
                               K_GLOBAL);
      }
    }

    for (int c = 0; c < COMPLEX; c++) {
      for (int n = 0; n < N_TILES; n++) {
        int k_index = k * K_PER_WMMA;
        wmma::load_matrix_sync(b[c][n],
                               &B[c][blockN * N_PER_BLOCK + warpN * N_PER_WARP +
                                     n * N_PER_WMMA][k_index],
                               K_GLOBAL);
      }
    }

    // step 1 and 2
    for (int m = 0; m < M_TILES; m++) {
      for (int n = 0; n < N_TILES; n++) {
        wmma::mma_sync(sum[REAL][m][n], a[REAL][m], b[REAL][n],
                       sum[REAL][m][n]);
        wmma::mma_sync(sum[IMAG][m][n], a[REAL][m], b[IMAG][n],
                       sum[IMAG][m][n]);
      }
    }

    // step 3
    __syncwarp();
    for (int n = 0; n < N_TILES; n++) {
      for (auto &element : b[IMAG][n].x) {
        element = -element;
      }
    }
    __syncwarp();

    // step 4 and 5
    for (int m = 0; m < M_TILES; m++) {
      for (int n = 0; n < N_TILES; n++) {
        wmma::mma_sync(sum[REAL][m][n], a[IMAG][m], b[IMAG][n],
                       sum[REAL][m][n]);
        wmma::mma_sync(sum[IMAG][m][n], a[IMAG][m], b[REAL][n],
                       sum[IMAG][m][n]);
      }
    }
  }

  // store the result to global memory
  for (int c = 0; c < COMPLEX; c++) {
    for (int m = 0; m < M_TILES; m++) {
      for (int n = 0; n < N_TILES; n++) {
        Tout *c_ptr =
            &C[c][blockM * M_PER_BLOCK + warpM * M_PER_WARP + m * M_PER_WMMA]
              [blockN * N_PER_BLOCK + warpN * N_PER_WARP + n * N_PER_WMMA];
        wmma::store_matrix_sync(c_ptr, sum[c][m][n], N_GLOBAL,
                                wmma::mem_row_major);
      }
    }
  }
}

extern "C" __global__ void wmma_complex_gemm_opt(C_t C, const A_opt_t A,
                                                 const B_opt_t B) {
  const unsigned blockN = blockIdx.x;
  const unsigned blockM = blockIdx.y;
  const unsigned warpN = threadIdx.y;
  const unsigned warpM = threadIdx.z;

  constexpr unsigned num_threads = block_size_x * block_size_y * block_size_z;
  const unsigned tid = threadIdx.x + threadIdx.y * block_size_x +
                       threadIdx.z * block_size_x * block_size_y;

  // number of tiles processed by each warp
  constexpr unsigned M_TILES = M_PER_WARP / M_PER_WMMA;
  constexpr unsigned N_TILES = N_PER_WARP / N_PER_WMMA;
  constexpr unsigned K_TILES = K_GLOBAL / K_PER_WMMA;

  // initialize accumulator fragments to zero
  wmma::fragment<wmma::accumulator, M_PER_WMMA, N_PER_WMMA, K_PER_WMMA, Tout>
      sum[COMPLEX][M_TILES][N_TILES];
  for (int c = 0; c < COMPLEX; c++) {
    for (int m = 0; m < M_TILES; m++) {
      for (int n = 0; n < N_TILES; n++) {
        wmma::fill_fragment(sum[c][m][n], 0);
      }
    }
  }

  // shared memory buffers for partial A and B matrix. Several buffers to allow
  // for async operations: copy next submatrix to shared memory while computing
  // current submatrix
  __shared__ Tin A_s[NBUFFER][COMPLEX][M_PER_BLOCK][K_PER_WMMA];
  __shared__ Tin B_s[NBUFFER][COMPLEX][N_PER_BLOCK][K_PER_WMMA];

  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

  for (unsigned k = 0, k_buf = 0; k < K_TILES; k++) {

    // declare input fragments for A and B matrices
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
          &A_s[k_buf % NBUFFER][0][0], &A[blockM][k_buf][0][0], pipe, tid);
      copy_async<sizeof(B_s[0]), num_threads>(
          &B_s[k_buf % NBUFFER][0][0], &B[blockN][k_buf][0][0], pipe, tid);
      pipe.producer_commit();
    }

    // NBUFFER copy operations have been started
    // the oldest one needs to be finished before we can start computation on
    // that data This corresponds to (NBUFFER - 1) copy operations ago so that
    // is the one we need to wait for
    cuda::pipeline_consumer_wait_prior<NBUFFER - 1>(pipe);
    __syncthreads(); // not sure if this is needed, perhaps already handled by
                     // the pipe.wait_prior

    // load A matrix from shared memory
    for (int c = 0; c < COMPLEX; c++) {
      for (int m = 0; m < M_TILES; m++) {
        wmma::load_matrix_sync(
            a[c][m],
            &A_s[k % NBUFFER][c][warpM * M_PER_WARP + m * M_PER_WMMA][0],
            K_PER_WMMA);
      }
    }

    // load B matrix from shared memory
    for (int c = 0; c < COMPLEX; c++) {
      for (int n = 0; n < N_TILES; n++) {
        wmma::load_matrix_sync(
            b[c][n],
            &B_s[k % NBUFFER][c][warpN * N_PER_WARP + n * N_PER_WMMA][0],
            K_PER_WMMA);
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

    // steps 1 and 2
    for (int m = 0; m < M_TILES; m++) {
      for (int n = 0; n < N_TILES; n++) {
        wmma::mma_sync(sum[REAL][m][n], a[REAL][m], b[REAL][n],
                       sum[REAL][m][n]);
        wmma::mma_sync(sum[IMAG][m][n], a[REAL][m], b[IMAG][n],
                       sum[IMAG][m][n]);
      }
    }

    // step 3
    __syncwarp();
    for (int n = 0; n < N_TILES; n++) {
      for (auto &element : b[IMAG][n].x) {
        element = -element;
      }
    }
    __syncwarp();

    // step 4 and 5
    for (int m = 0; m < M_TILES; m++) {
      for (int n = 0; n < N_TILES; n++) {
        wmma::mma_sync(sum[REAL][m][n], a[IMAG][m], b[IMAG][n],
                       sum[REAL][m][n]);
        wmma::mma_sync(sum[IMAG][m][n], a[IMAG][m], b[REAL][n],
                       sum[IMAG][m][n]);
      }
    }

    pipe.consumer_release();

    __syncthreads();
  }

  // store the result to global memory
  for (int c = 0; c < COMPLEX; c++) {
    for (int m = 0; m < M_TILES; m++) {
      for (int n = 0; n < N_TILES; n++) {
        Tout *c_ptr =
            &C[c][blockM * M_PER_BLOCK + warpM * M_PER_WARP + m * M_PER_WMMA]
              [blockN * N_PER_BLOCK + warpN * N_PER_WARP + n * N_PER_WMMA];
        wmma::store_matrix_sync(c_ptr, sum[c][m][n], N_GLOBAL,
                                wmma::mem_row_major);
      }
    }
  }
}
