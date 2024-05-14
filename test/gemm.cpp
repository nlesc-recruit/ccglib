#include <complex>
#include <iostream>
#include <type_traits>

#include <cuda_fp16.h>
#include <math.h>
#include <omp.h>

#include <xtensor/xadapt.hpp>
#include <xtensor/xtensor.hpp>

#include <cudawrappers/cu.hpp>
#include <cudawrappers/nvrtc.hpp>

#include "mma/GEMM.h"
#include "reference/GEMM.h"
#include "transpose/Transpose.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "verify.h"

#ifndef COMPLEX
#define COMPLEX 2
#endif

namespace ccglib::test {

class ComplexGemmTestFixture {
public:
  ComplexGemmTestFixture() {}

private:
  template <typename T>
  void init_input_matrices(T *a, T *b, const size_t bytes_a,
                           const size_t bytes_b) {
    // fill a and b with random values (fixed seed), initalize c to zero
    // Note: only works for T=half, should use e.g. KernelFloat library to
    // more easily support other types
    static_assert(std::is_same_v<T, __half>, "Input data type must be half");
    unsigned int seed = 0;
    const float scale = 1.0f;
    for (int idx = 0; idx < bytes_a / sizeof(T); idx++) {
      a[idx] = __float2half(2.0f * scale *
                                (static_cast<float>(rand_r(&seed)) / RAND_MAX) -
                            scale);
    }
    for (int idx = 0; idx < bytes_b / sizeof(T); idx++) {
      b[idx] = __float2half(2.0f * scale *
                                (static_cast<float>(rand_r(&seed)) / RAND_MAX) -
                            scale);
    }
  }

protected:
  void complex_gemm_basic() {
    cu::init();
    cu::Device device(0);
    cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, device);
    cu::Stream stream;

    // kernel settings
    const int m_per_block = ccglib::mma::GEMM::kMPerBlock;
    const int n_per_block = ccglib::mma::GEMM::kNPerBlock;
    const int k_per_wmma = ccglib::mma::GEMM::kKPerWMMA;

    // data size and type
    const int global_m = m_per_block;    // must be multiple of m_per_block
    const int global_n = n_per_block;    // must be multiple of n_per_block
    const int global_k = 4 * k_per_wmma; // must be multiple of k_per_wmma
    const int batch_size = 16;

    using Tin = half;
    using Tout = float;

    const size_t nr_input_bits = sizeof(Tin) * 8;
    const size_t nr_output_bits = sizeof(Tout) * 8;

    const size_t bytes_a =
        sizeof(Tin) * batch_size * COMPLEX * global_m * global_k;
    const size_t bytes_b =
        sizeof(Tin) * batch_size * COMPLEX * global_n * global_k;
    const size_t bytes_c =
        sizeof(Tout) * batch_size * COMPLEX * global_m * global_n;

    // initalize host memory
    cu::HostMemory h_a(bytes_a);
    cu::HostMemory h_b(bytes_b);
    cu::HostMemory h_c(bytes_c);
    init_input_matrices(static_cast<Tin *>(h_a), static_cast<Tin *>(h_b),
                        bytes_a, bytes_b);

    // Allocate device memory for input data
    cu::DeviceMemory d_a(bytes_a);
    cu::DeviceMemory d_b(bytes_b);

    // Transfer the input data
    stream.memcpyHtoDAsync(d_a, h_a, bytes_a);
    stream.memcpyHtoDAsync(d_b, h_b, bytes_b);

    // allocate device memory for output data and initialize to zero
    cu::DeviceMemory d_c(bytes_c);
    d_c.zero(bytes_c);

    ccglib::mma::GEMM gemm_mma(batch_size, global_m, global_k, global_n,
                               nr_input_bits, nr_output_bits, device, stream,
                               ccglib::mma::GEMM::basic);

    // run the GEMM kernel
    gemm_mma.run(d_a, d_b, d_c);

    // copy C to host
    stream.memcpyDtoHAsync(h_c, d_c, bytes_c);
    stream.synchronize();

    // verify output
    verify<Tin, Tout, batch_size, global_m, global_n, global_k>(
        static_cast<const Tin *>(h_a), static_cast<const Tin *>(h_b),
        static_cast<Tout *>(h_c));
  }

  void complex_gemm_opt() {
    cu::init();
    cu::Device device(0);
    cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, device);
    cu::Stream stream;

    // kernel settings
    const int m_per_block = ccglib::mma::GEMM::kMPerBlock;
    const int n_per_block = ccglib::mma::GEMM::kNPerBlock;
    const int k_per_wmma = ccglib::mma::GEMM::kKPerWMMA;

    // data size and type
    const int global_m = m_per_block;    // must be multiple of m_per_block
    const int global_n = n_per_block;    // must be multiple of n_per_block
    const int global_k = 4 * k_per_wmma; // must be multiple of k_per_wmma
    const int batch_size = 16;

    using Tin = half;
    using Tout = float;

    const unsigned int nr_input_bits = sizeof(Tin) * 8;
    const unsigned int nr_output_bits = sizeof(Tout) * 8;

    const size_t bytes_a =
        sizeof(Tin) * batch_size * COMPLEX * global_m * global_k;
    const size_t bytes_b =
        sizeof(Tin) * batch_size * COMPLEX * global_n * global_k;
    const size_t bytes_c =
        sizeof(Tout) * batch_size * COMPLEX * global_m * global_n;

    // initalize host memory
    cu::HostMemory h_a(bytes_a);
    cu::HostMemory h_b(bytes_b);
    cu::HostMemory h_c(bytes_c);
    init_input_matrices(static_cast<Tin *>(h_a), static_cast<Tin *>(h_b),
                        bytes_a, bytes_b);

    // Allocate device memory for transposed input data
    cu::DeviceMemory d_a_trans(bytes_a);
    cu::DeviceMemory d_b_trans(bytes_b);

    // Transpose A
    ccglib::transpose::Transpose transpose_a(batch_size, global_m, global_k,
                                             m_per_block, k_per_wmma,
                                             nr_input_bits, device, stream);
    transpose_a.run(h_a, d_a_trans);

    // Transpose B
    ccglib::transpose::Transpose transpose_b(batch_size, global_n, global_k,
                                             n_per_block, k_per_wmma,
                                             nr_input_bits, device, stream);
    transpose_b.run(h_b, d_b_trans);

    // allocate device memory for output data and initialize to zero
    cu::DeviceMemory d_c(bytes_c);
    d_c.zero(bytes_c);

    ccglib::mma::GEMM gemm_mma(batch_size, global_m, global_k, global_n,
                               nr_input_bits, nr_output_bits, device, stream,
                               ccglib::mma::GEMM::opt);

    // run the GEMM kernel
    gemm_mma.run(d_a_trans, d_b_trans, d_c);

    // copy C to host
    stream.memcpyDtoHAsync(h_c, d_c, bytes_c);
    stream.synchronize();

    // verify output
    verify<Tin, Tout, batch_size, global_m, global_n, global_k>(
        static_cast<const Tin *>(h_a), static_cast<const Tin *>(h_b),
        static_cast<Tout *>(h_c));
  }
};

TEST_CASE_METHOD(ComplexGemmTestFixture, "Complex GEMM Test",
                 "[complex-gemm-test-basic]") {
  ComplexGemmTestFixture::complex_gemm_basic();
}

TEST_CASE_METHOD(ComplexGemmTestFixture, "Complex GEMM Test",
                 "[complex-gemm-test-opt]") {
  ComplexGemmTestFixture::complex_gemm_opt();
}

} // namespace ccglib::test
