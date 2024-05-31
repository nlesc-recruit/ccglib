#include <complex>
#include <iostream>
#include <type_traits>

#include <cuda_fp16.h>
#include <math.h>
#include <omp.h>

#include <cudawrappers/cu.hpp>
#include <cudawrappers/nvrtc.hpp>

#include <ccglib/gemm/mma.h>
#include <ccglib/gemm/reference.h>
#include <ccglib/transpose/transpose.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "verify.h"

#ifndef COMPLEX
#define COMPLEX 2
#endif

using Tin = half;
using Tout = float;

namespace ccglib::test {

class ComplexGemmTestFixture {
public:
  ComplexGemmTestFixture() {
    cu::init();
    device_ = std::make_unique<cu::Device>(0);
    context_ =
        std::make_unique<cu::Context>(CU_CTX_SCHED_BLOCKING_SYNC, *device_);
    stream_ = std::make_unique<cu::Stream>();
    kernel_ = std::make_unique<ccglib::mma::Kernel>(ccglib::mma::float16,
                                                    ccglib::mma::opt);

    const ccglib::mma::Kernel::Parameters parameters = kernel_->GetParameters();

    global_m_ = parameters.m_per_block;    // must be multiple of m_per_block
    global_n_ = parameters.n_per_block;    // must be multiple of n_per_block
    global_k_ = 4 * parameters.k_per_wmma; // must be multiple of k_per_wmma

    bytes_a_ = sizeof(Tin) * kBatchSize * COMPLEX * global_m_ * global_k_;
    bytes_b_ = sizeof(Tin) * kBatchSize * COMPLEX * global_n_ * global_k_;
    bytes_c_ = sizeof(Tout) * kBatchSize * COMPLEX * global_m_ * global_n_;
  }

private:
  std::unique_ptr<cu::Device> device_;
  std::unique_ptr<cu::Context> context_;
  std::unique_ptr<cu::Stream> stream_;
  std::unique_ptr<ccglib::mma::Kernel> kernel_;

  std::unique_ptr<cu::HostMemory> h_a_;
  std::unique_ptr<cu::HostMemory> h_b_;
  std::unique_ptr<cu::HostMemory> h_c_;

  std::unique_ptr<cu::DeviceMemory> d_a_;
  std::unique_ptr<cu::DeviceMemory> d_b_;
  std::unique_ptr<cu::DeviceMemory> d_c_;

  // data size and type
  size_t global_m_;
  size_t global_n_;
  size_t global_k_;
  const size_t kBatchSize = 16;

  const size_t kNrInputBits = sizeof(Tin) * 8;

  size_t bytes_a_;
  size_t bytes_b_;
  size_t bytes_c_;

  template <typename T> void init_input_matrices(T *a, T *b) {
    // fill a and b with random values (fixed seed), initalize c to zero
    // Note: only works for T=half, should use e.g. KernelFloat library to
    // more easily support other types
    static_assert(std::is_same_v<T, __half>, "Input data type must be half");
    unsigned int seed = 0;
    const float scale = 1.0f;
    for (int idx = 0; idx < bytes_a_ / sizeof(T); idx++) {
      a[idx] = __float2half(2.0f * scale *
                                (static_cast<float>(rand_r(&seed)) / RAND_MAX) -
                            scale);
    }
    for (int idx = 0; idx < bytes_b_ / sizeof(T); idx++) {
      b[idx] = __float2half(2.0f * scale *
                                (static_cast<float>(rand_r(&seed)) / RAND_MAX) -
                            scale);
    }
  }

  void initialize_memory() {
    // initalize host memory
    h_a_ = std::make_unique<cu::HostMemory>(bytes_a_);
    h_b_ = std::make_unique<cu::HostMemory>(bytes_b_);
    h_c_ = std::make_unique<cu::HostMemory>(bytes_c_);
    init_input_matrices(static_cast<Tin *>(*h_a_), static_cast<Tin *>(*h_b_));

    // Allocate device memory for input data
    d_a_ = std::make_unique<cu::DeviceMemory>(bytes_a_);
    d_b_ = std::make_unique<cu::DeviceMemory>(bytes_b_);

    // Transfer the input data
    stream_->memcpyHtoDAsync(*d_a_, *h_a_, bytes_a_);
    stream_->memcpyHtoDAsync(*d_b_, *h_b_, bytes_b_);

    // allocate device memory for output data and initialize to zero
    d_c_ = std::make_unique<cu::DeviceMemory>(bytes_c_);
    d_c_->zero(bytes_c_);
  }

  void verify_output() {
    // copy C to host
    stream_->memcpyDtoHAsync(*h_c_, *d_c_, bytes_c_);
    stream_->synchronize();

    // verify output
    verify<Tin, Tout>(static_cast<const Tin *>(*h_a_),
                      static_cast<const Tin *>(*h_b_),
                      static_cast<Tout *>(*h_c_), kBatchSize, global_m_,
                      global_n_, global_k_);
  }

protected:
  void complex_gemm_basic() {
    initialize_memory();

    ccglib::mma::GEMM gemm_mma(kBatchSize, global_m_, global_n_, global_k_,
                               kNrInputBits, *device_, *stream_,
                               ccglib::mma::float16, ccglib::mma::basic);

    gemm_mma.run(*d_a_, *d_b_, *d_c_);

    verify_output();
  }

  void complex_gemm_opt() {
    initialize_memory();

    // Allocate device memory for transposed input data
    cu::DeviceMemory d_a_trans(bytes_a_);
    cu::DeviceMemory d_b_trans(bytes_b_);

    const ccglib::mma::Kernel::Parameters parameters = kernel_->GetParameters();

    // Transpose A
    ccglib::transpose::Transpose transpose_a(
        kBatchSize, global_m_, global_k_, parameters.m_per_block,
        parameters.k_per_wmma, kNrInputBits, *device_, *stream_);
    transpose_a.run(*h_a_, d_a_trans);

    // Transpose B
    ccglib::transpose::Transpose transpose_b(
        kBatchSize, global_n_, global_k_, parameters.n_per_block,
        parameters.k_per_wmma, kNrInputBits, *device_, *stream_);
    transpose_b.run(*h_b_, d_b_trans);

    ccglib::mma::GEMM gemm_mma(kBatchSize, global_m_, global_k_, global_n_,
                               kNrInputBits, *device_, *stream_,
                               ccglib::mma::float16, ccglib::mma::opt);

    gemm_mma.run(d_a_trans, d_b_trans, *d_c_);

    verify_output();
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
