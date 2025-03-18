#if defined(__HIP_PLATFORM_AMD__)
#include <rocwmma/rocwmma.hpp>
namespace wmma = rocwmma;
#include "sync_copies.h"
#else
#include "async_copies.h"
#include "wmma_extension.h"
#include <cuda/pipeline>
#include <mma.h>
using namespace nvcuda;
#endif

#include "ccglib/fp16.h"
#include "matrix_operations.h"
#include "type_selector.h"

// Check memory layout of A and B matrix
#ifdef A_COL_MAJOR
#error "float GEMM does not currently support col-major A matrix"
#endif
#ifdef B_ROW_MAJOR
#error "float GEMM does not currently support row-major B matrix"
#endif

// data layout for optimal transfer to shared memory
using A_opt_t =
    Tin[BATCH_SIZE][M_GLOBAL_PADDED / M_PER_BLOCK][K_GLOBAL_PADDED / K_PER_WMMA]
       [COMPLEX][M_PER_BLOCK][K_PER_WMMA];
using B_opt_t =
    Tin[BATCH_SIZE][N_GLOBAL_PADDED / N_PER_BLOCK][K_GLOBAL_PADDED / K_PER_WMMA]
       [COMPLEX][N_PER_BLOCK][K_PER_WMMA];

extern "C" __global__ void wmma_complex_gemm_basic(C_t C, const A_t A,
                                                   const B_t B) {
  const size_t batch = blockIdx.z;
  const size_t blockN = blockIdx.x;
  const size_t blockM = blockIdx.y;
  const size_t warpN = threadIdx.y;
  const size_t warpM = threadIdx.z;

  constexpr size_t M_TILES = M_PER_WARP / M_PER_WMMA;
  constexpr size_t N_TILES = N_PER_WARP / N_PER_WMMA;
  constexpr size_t K_TILES = K_GLOBAL_PADDED / ACCUMULATOR_K_PER_WMMA;

  Accumulator_t sum[COMPLEX][M_TILES][N_TILES];
  for (size_t c = 0; c < COMPLEX; c++) {
    for (size_t m = 0; m < M_TILES; m++) {
      for (size_t n = 0; n < N_TILES; n++) {
        wmma::fill_fragment(sum[c][m][n], static_cast<Tout>(.0f));
      }
    }
  }

#if K_IS_PADDED || M_IS_PADDED
  __shared__ Tin A_s[M_PER_BLOCK / M_PER_WARP][N_PER_BLOCK / N_PER_WARP]
                    [M_PER_WMMA][K_PER_WMMA];
#endif
#if K_IS_PADDED || N_IS_PADDED
  __shared__ Tin B_s[M_PER_BLOCK / M_PER_WARP][N_PER_BLOCK / N_PER_WARP]
                    [N_PER_WMMA][K_PER_WMMA];
#endif

#if REQUIRES_SHARED_MEMORY
  __shared__ Tshared C_s[M_PER_BLOCK / M_PER_WARP][N_PER_BLOCK / N_PER_WARP]
                        [M_PER_WMMA][N_PER_WMMA];
#endif

  for (size_t k = 0; k < K_TILES; k++) {
    // declare input fragments
    wmma::fragment<wmma::matrix_a, M_PER_WMMA, N_PER_WMMA, K_PER_WMMA, Ttc,
                   wmma::row_major>
        a[COMPLEX][M_TILES];
    wmma::fragment<wmma::matrix_b, M_PER_WMMA, N_PER_WMMA, K_PER_WMMA, Ttc,
                   wmma::col_major>
        b[COMPLEX][N_TILES];

    // load matrices from global memory
    // float is implicitly converted to tf32. No conversion is done in half
    // precision mode.
    for (size_t c = 0; c < COMPLEX; c++) {
      for (size_t m = 0; m < M_TILES; m++) {
        size_t k_index = k * K_PER_WMMA;
#if K_IS_PADDED || M_IS_PADDED
        size_t m_index = global_idx_m(blockM, warpM, m);
        __syncwarp();
        for (size_t t = threadIdx.x; t < M_PER_WMMA * K_PER_WMMA;
             t += WARP_SIZE) {
          size_t i = t / K_PER_WMMA;
          size_t j = t % K_PER_WMMA;
          // Transfer the submatrix; pad the out-of-bounds values with zeros
          if (m_index + i < M_GLOBAL && k_index + j < K_GLOBAL) {
            A_s[warpM][warpN][i][j] = A[batch][c][m_index + i][k_index + j];
          } else {
            A_s[warpM][warpN][i][j] = 0;
          }
        }
        __syncwarp();
        wmma::load_matrix_sync(a[c][m], &A_s[warpM][warpN][0][0], K_PER_WMMA);
#else
        wmma::load_matrix_sync(
            a[c][m], &A[batch][c][global_idx_m(blockM, warpM, m)][k_index],
            K_GLOBAL);
#endif
      }
    }

    for (size_t c = 0; c < COMPLEX; c++) {
      for (size_t n = 0; n < N_TILES; n++) {
        size_t k_index = k * K_PER_WMMA;
#if K_IS_PADDED || N_IS_PADDED
        size_t n_index = global_idx_n(blockN, warpN, n);
        __syncwarp();
        for (size_t t = threadIdx.x; t < N_PER_WMMA * K_PER_WMMA;
             t += WARP_SIZE) {
          size_t i = t / K_PER_WMMA;
          size_t j = t % K_PER_WMMA;
          // store the submatrix, padded values are set to zero
          if (n_index + i < N_GLOBAL && k_index + j < K_GLOBAL) {
            B_s[warpM][warpN][i][j] = B[batch][c][n_index + i][k_index + j];
          } else {
            B_s[warpM][warpN][i][j] = 0;
          }
        }
        __syncwarp();
        wmma::load_matrix_sync(b[c][n], &B_s[warpM][warpN][0][0], K_PER_WMMA);
#else
        wmma::load_matrix_sync(
            b[c][n], &B[batch][c][global_idx_n(blockN, warpN, n)][k_index],
            K_GLOBAL);
#endif
      }
    }

    // step 1 and 2
    for (size_t m = 0; m < M_TILES; m++) {
      for (size_t n = 0; n < N_TILES; n++) {
        wmma::mma_sync(sum[REAL][m][n], a[REAL][m], b[REAL][n],
                       sum[REAL][m][n]);
        wmma::mma_sync(sum[IMAG][m][n], a[REAL][m], b[IMAG][n],
                       sum[IMAG][m][n]);
      }
    }

    // step 3
    __syncwarp();
    for (size_t n = 0; n < N_TILES; n++) {
      for (size_t element = 0; element < b[IMAG][n].num_elements; element++) {
        b[IMAG][n].x[element] = -b[IMAG][n].x[element];
      }
    }
    __syncwarp();

    // step 4 and 5
    for (size_t m = 0; m < M_TILES; m++) {
      for (size_t n = 0; n < N_TILES; n++) {
        wmma::mma_sync(sum[REAL][m][n], a[IMAG][m], b[IMAG][n],
                       sum[REAL][m][n]);
        wmma::mma_sync(sum[IMAG][m][n], a[IMAG][m], b[REAL][n],
                       sum[IMAG][m][n]);
      }
    }
  } // k

// store the result to global memory
#if REQUIRES_SHARED_MEMORY
  store_matrix_padded(sum, C, C_s, batch, blockM, warpM, blockN, warpN, M_TILES,
                      N_TILES);
#else
  store_matrix(sum, C, batch, blockM, warpM, blockN, warpN);
#endif
}

// Optimized GEMM kernel
// On NVIDIA GPUs, multiple shared memory buffers and asynchronous memory
// copies are used. On AMD GPUs, one shared memory buffer is used.
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
  constexpr size_t K_TILES = K_GLOBAL_PADDED / K_PER_WMMA;

  // initialize sum fragments to zero
  Accumulator_t sum[COMPLEX][M_TILES][N_TILES];
  for (size_t c = 0; c < COMPLEX; c++) {
    for (size_t m = 0; m < M_TILES; m++) {
      for (size_t n = 0; n < N_TILES; n++) {
        wmma::fill_fragment(sum[c][m][n], static_cast<Tout>(.0f));
      }
    }
  }

  // shared memory buffers for partial A and B matrix. Several buffers
  // to allow for async operations: copy next submatrix to shared memory while
  // computing current submatrix
  // To save shared memory, the C matrix reuses the same shared memory in
  // case of padding.
  constexpr size_t A_s_size = NBUFFER * COMPLEX * M_PER_BLOCK * K_PER_WMMA;
  constexpr size_t B_s_size = NBUFFER * COMPLEX * N_PER_BLOCK * K_PER_WMMA;
  __shared__ Tin shmem[A_s_size + B_s_size];
  typedef Tin(*A_s_t)[COMPLEX][M_PER_BLOCK / M_PER_WARP][M_TILES][M_PER_WMMA]
                     [K_PER_WMMA];
  typedef Tin(*B_s_t)[COMPLEX][N_PER_BLOCK / N_PER_WARP][N_TILES][N_PER_WMMA]
                     [K_PER_WMMA];
  A_s_t A_s = reinterpret_cast<A_s_t>(&shmem[0]);
  B_s_t B_s = reinterpret_cast<B_s_t>(&shmem[A_s_size]);
  typedef Tshared(*C_s_t)[N_PER_BLOCK / N_PER_WARP][M_PER_WMMA][N_PER_WMMA];
#if REQUIRES_SHARED_MEMORY
  C_s_t C_s = reinterpret_cast<C_s_t>(&shmem[0]);
#endif

#if !defined(__HIP_PLATFORM_AMD__)
  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
#endif

  for (size_t k = 0, k_buf = 0; k < K_TILES; k++) {

    // declare input fragments for A and B matrices
    wmma::fragment<wmma::matrix_a, M_PER_WMMA, N_PER_WMMA, K_PER_WMMA, Ttc,
                   wmma::row_major>
        a[COMPLEX][M_TILES];
    wmma::fragment<wmma::matrix_b, M_PER_WMMA, N_PER_WMMA, K_PER_WMMA, Ttc,
                   wmma::col_major>
        b[COMPLEX][N_TILES];

    // copy next data to smem
#if defined(__HIP_PLATFORM_AMD__)
    copy_sync<int4, sizeof(A_s[0]), num_threads>(
        &A_s[0][0][0], &A[batch][blockM][k][0][0][0], tid);
    copy_sync<int4, sizeof(B_s[0]), num_threads>(
        &B_s[0][0][0], &B[batch][blockN][k][0][0][0], tid);
#else
    for (; k_buf < K_TILES && k_buf < (k + NBUFFER); k_buf++) {
      pipe.producer_acquire();
      copy_async<sizeof(A_s[0]), num_threads>(
          &A_s[k_buf % NBUFFER][0][0][0][0][0], &A[batch][blockM][k_buf][0][0],
          pipe, tid);
      copy_async<sizeof(B_s[0]), num_threads>(
          &B_s[k_buf % NBUFFER][0][0][0][0][0], &B[batch][blockN][k_buf][0][0],
          pipe, tid);
      pipe.producer_commit();
    }

    // NBUFFER copy operations have been started
    // the oldest one needs to be finished before we can start computation on
    // that data This corresponds to (NBUFFER - 1) copy operations ago so that
    // is the one we need to wait for
    pipe.consumer_wait();

#endif
    // Synchronize threads before loading the fragments from shared memory
    __syncthreads();

    // load A matrix from shared memory
    // float is implicitly converted to tf32. No conversion is done in half
    // precision mode.
    for (size_t c = 0; c < COMPLEX; c++) {
      for (size_t m = 0; m < M_TILES; m++) {
        wmma::load_matrix_sync(a[c][m], &A_s[k % NBUFFER][c][warpM][m][0][0],
                               K_PER_WMMA);
      }
    }

    // load B matrix from shared memory
    // float is implicitly converted to tf32. No conversion is done in half
    // precision mode.
    for (size_t c = 0; c < COMPLEX; c++) {
      for (size_t n = 0; n < N_TILES; n++) {
        wmma::load_matrix_sync(b[c][n], &B_s[k % NBUFFER][c][warpN][n][0][0],
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
    for (size_t m = 0; m < M_TILES; m++) {
      for (size_t n = 0; n < N_TILES; n++) {
        wmma::mma_sync(sum[REAL][m][n], a[REAL][m], b[REAL][n],
                       sum[REAL][m][n]);
        wmma::mma_sync(sum[IMAG][m][n], a[REAL][m], b[IMAG][n],
                       sum[IMAG][m][n]);
      }
    }

    // step 3
    __syncwarp();
    for (size_t n = 0; n < N_TILES; n++) {
      for (size_t element = 0; element < b[IMAG][n].num_elements; element++) {
        b[IMAG][n].x[element] = -b[IMAG][n].x[element];
      }
    }
    __syncwarp();

    // step 4 and 5
    for (size_t m = 0; m < M_TILES; m++) {
      for (size_t n = 0; n < N_TILES; n++) {
        wmma::mma_sync(sum[REAL][m][n], a[IMAG][m], b[IMAG][n],
                       sum[REAL][m][n]);
        wmma::mma_sync(sum[IMAG][m][n], a[IMAG][m], b[REAL][n],
                       sum[IMAG][m][n]);
      }
    }

#if !defined(__HIP_PLATFORM_AMD__)
    pipe.consumer_release();
#endif

    __syncthreads();
  } // k

// store the result to global memory
#if REQUIRES_SHARED_MEMORY
  store_matrix_padded(sum, C, C_s, batch, blockM, warpM, blockN, warpN, M_TILES,
                      N_TILES);
#else
  store_matrix(sum, C, batch, blockM, warpM, blockN, warpN);
#endif
}
