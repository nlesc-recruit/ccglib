#include <complex>
#include <cuda_fp16.h>
#include <iostream>
#include <math.h>
#include <omp.h>

#include <xtensor/xadapt.hpp>
#include <xtensor/xtensor.hpp>

#include <cudawrappers/cu.hpp>
#include <cudawrappers/nvrtc.hpp>

#include "mma/GEMM.h"
#include "reference/GEMM.h"
#include "transpose/Transpose.h"

#ifndef COMPLEX
#define COMPLEX 2
#endif

extern const char _binary_kernels_transpose_kernel_cu_start,
    _binary_kernels_transpose_kernel_cu_end;

template <typename Tin, typename Tout, unsigned M, unsigned N, unsigned K>
void verify(const Tin *a, const Tin *b, const Tout *c) {
  const std::array<size_t, 3> a_shape = {2, M, K};
  const std::array<size_t, 3> b_shape = {2, N, K};
  const std::array<size_t, 3> c_shape = {2, M, N};

  const size_t a_size = 2 * M * K;
  const size_t b_size = 2 * N * K;
  const size_t c_size = 2 * M * N;

  auto a_view = xt::adapt(a, a_size, xt::no_ownership(), a_shape);
  auto b_view = xt::adapt(b, b_size, xt::no_ownership(), b_shape);
  auto c_view = xt::adapt(c, c_size, xt::no_ownership(), c_shape);

  xt::xtensor<Tout, 3> c_ref = xt::zeros_like(c_view);

  ccglib::reference::GEMM gemm;
  gemm.run(a, b, c_ref.data(), M, N, K);

  std::cout << "Verifying output" << std::endl;
  const int max_errs = 10;
  int errs = 0;
  for (unsigned m = 0; m < M; m++) {
    for (unsigned n = 0; n < N; n++) {
      if (errs >= max_errs) {
        break;
      }
      std::complex<Tout> ref(c_ref(0, m, n), c_ref(1, m, n));
      std::complex<Tout> tst(c_view(0, m, n), c_view(1, m, n));
      if (std::abs(ref - tst) > 1 && errs < max_errs) {
        std::cout << "Failed at m=" << m << ", n=" << n;
        std::cout << ", expected " << ref << ", found " << tst << std::endl;
        errs++;
      }
    }
  }
  if (errs == 0) {
    std::cout << "Result ok" << std::endl;
  }
}

int main() {
  std::cout << "Beamform main" << std::endl;

  cu::init();
  cu::Device device(0);
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, device);
  cu::Stream stream;

  // kernel settings
  const int beams_per_block = ccglib::mma::GEMM::kBeamsPerBlock;
  const int frames_per_block = ccglib::mma::GEMM::kFramesPerBlock;
  const int samples_per_wmma = ccglib::mma::GEMM::kSamplesPerWMMA;

  // data size and type, sizes match CUBE test data
  const int beams = 10240;  // must be multiple of beams_per_block
  const int frames = 1024;  // must be multiple of frames_per_block
  const int samples = 7808; // must be multiple of samples_per_wmma

  using Tin = half;
  using Tout = float;

  const unsigned int nr_input_bits = sizeof(Tin) * 8;
  const unsigned int nr_output_bits = sizeof(Tout) * 8;

  // data types for matrices
  // A and B are transposed to an optimal format for the GEMM
  using A_t = Tin[COMPLEX][beams][samples];
  using B_t = Tin[COMPLEX][frames][samples];
  using A_trans_t = Tin[beams / beams_per_block][samples / samples_per_wmma]
                       [COMPLEX][beams_per_block][samples_per_wmma];
  using B_trans_t = Tin[frames / frames_per_block][samples / samples_per_wmma]
                       [COMPLEX][frames_per_block][samples_per_wmma];
  using C_t = Tout[COMPLEX][beams][frames];

  const size_t bytes_a = sizeof(A_t);
  const size_t bytes_b = sizeof(B_t);
  const size_t bytes_c = sizeof(C_t);

  // initalize host memory
  cu::HostMemory h_a(bytes_a);
  cu::HostMemory h_b(bytes_b);
  cu::HostMemory h_c(bytes_c);

  A_t *a = static_cast<A_t *>(h_a);
  B_t *b = static_cast<B_t *>(h_b);
  C_t *c = static_cast<C_t *>(h_c);

  // fill a and b with random values (fixed seed), initalize c to zero
  // Note: only works for Tin=half, should use e.g. KernelFloat library to
  // more easily support other types
  srand(42);
  for (int idx = 0; idx < bytes_a / sizeof(Tin); idx++) {
    static_cast<Tin *>(h_a)[idx] =
        __float2half(16 * ((float)rand() / RAND_MAX) - 8);
  }
  for (int idx = 0; idx < bytes_b / sizeof(Tin); idx++) {
    static_cast<Tin *>(h_b)[idx] =
        __float2half(16 * ((float)rand() / RAND_MAX) - 8);
  }

  // We start with a transpose kernel to get the data in the right shape for the
  // GEMM. The original data is allocated on the GPU in a subscope because we do
  // not need it anymore  after the transpose.
  // When DeviceMemory goes out of scope, the destructor will free the GPU
  // memory

  // Allocate device memory for transposed input data
  cu::DeviceMemory d_a_trans(bytes_a);
  cu::DeviceMemory d_b_trans(bytes_b);

  // Transpose A
  ccglib::transpose::Transpose transpose_a(beams, samples, beams_per_block,
                                           samples_per_wmma, nr_input_bits,
                                           device, stream);
  transpose_a.run(h_a, d_a_trans);

  // Transpose B
  ccglib::transpose::Transpose transpose_b(frames, samples, frames_per_block,
                                           samples_per_wmma, nr_input_bits,
                                           device, stream);
  transpose_b.run(h_b, d_b_trans);

  // allocate device memory for output data and initialize to zero
  cu::DeviceMemory d_c(bytes_c);
  d_c.zero(bytes_c);

  ccglib::mma::GEMM gemm_mma(beams, samples, frames, nr_input_bits,
                             nr_output_bits, device, stream);

  // run and time the GEMM kernel
  cu::Event start, end;
  start.record(stream);
  gemm_mma.run(d_a_trans, d_b_trans, d_c);
  end.record(stream);
  end.synchronize();

  float time = end.elapsedTime(start);
  std::cout << "Kernel took " << time << " ms" << std::endl;
  float tflops = 8ULL * 1e-9 * beams * frames * samples / time;
  std::cout << "TFLOPS: " << tflops << std::endl;

  // copy C to host
  stream.memcpyDtoHAsync(h_c, d_c, bytes_c);
  stream.synchronize();

  // verify output
  verify<Tin, Tout, beams, frames, samples>(reinterpret_cast<const Tin *>(a),
                                            reinterpret_cast<const Tin *>(b),
                                            reinterpret_cast<Tout *>(c));
  return 0;
}
