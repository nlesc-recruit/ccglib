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
  }

private:
  std::unique_ptr<cu::Device> device_;
  std::unique_ptr<cu::Context> context_;
  std::unique_ptr<cu::Stream> stream_;

  std::unique_ptr<cu::HostMemory> h_a_;
  std::unique_ptr<cu::HostMemory> h_b_;
  std::unique_ptr<cu::HostMemory> h_c_;

  std::unique_ptr<cu::DeviceMemory> d_a_;
  std::unique_ptr<cu::DeviceMemory> d_b_;
  std::unique_ptr<cu::DeviceMemory> d_c_;

  // kernel settings
  static const size_t kMPerBlock = ccglib::mma::GEMM::kMPerBlock;
  static const size_t kNPerBlock = ccglib::mma::GEMM::kNPerBlock;
  static const size_t kKPerWMMA = ccglib::mma::GEMM::kKPerWMMA;

  // data size and type
  static const size_t kGlobalM = kMPerBlock; // must be multiple of kMPerBlock
  static const size_t kGlobalN = kNPerBlock; // must be multiple of kNPerBlock
  static const size_t kGlobalK = 4 * kKPerWMMA; // must be multiple of kKPerWMMA
  static const size_t kBatchSize = 16;

  const size_t kNrInputBits = sizeof(Tin) * 8;
  const size_t kNrOutputBits = sizeof(Tout) * 8;

  const size_t kBytesA =
      sizeof(Tin) * kBatchSize * COMPLEX * kGlobalM * kGlobalK;
  const size_t kBytesB =
      sizeof(Tin) * kBatchSize * COMPLEX * kGlobalN * kGlobalK;
  const size_t kBytesC =
      sizeof(Tout) * kBatchSize * COMPLEX * kGlobalM * kGlobalN;

  template <typename T>
  void init_input_matrices(T *a, T *b, const size_t kBytesA,
                           const size_t kBytesB) {
    // fill a and b with random values (fixed seed), initalize c to zero
    // Note: only works for T=half, should use e.g. KernelFloat library to
    // more easily support other types
    static_assert(std::is_same_v<T, __half>, "Input data type must be half");
    unsigned int seed = 0;
    const float scale = 1.0f;
    for (int idx = 0; idx < kBytesA / sizeof(T); idx++) {
      a[idx] = __float2half(2.0f * scale *
                                (static_cast<float>(rand_r(&seed)) / RAND_MAX) -
                            scale);
    }
    for (int idx = 0; idx < kBytesB / sizeof(T); idx++) {
      b[idx] = __float2half(2.0f * scale *
                                (static_cast<float>(rand_r(&seed)) / RAND_MAX) -
                            scale);
    }
  }

  void initialize_memory() {
    // initalize host memory
    h_a_ = std::make_unique<cu::HostMemory>(kBytesA);
    h_b_ = std::make_unique<cu::HostMemory>(kBytesB);
    h_c_ = std::make_unique<cu::HostMemory>(kBytesC);
    init_input_matrices(static_cast<Tin *>(*h_a_), static_cast<Tin *>(*h_b_),
                        kBytesA, kBytesB);

    // Allocate device memory for input data
    d_a_ = std::make_unique<cu::DeviceMemory>(kBytesA);
    d_b_ = std::make_unique<cu::DeviceMemory>(kBytesB);

    // Transfer the input data
    stream_->memcpyHtoDAsync(*d_a_, *h_a_, kBytesA);
    stream_->memcpyHtoDAsync(*d_b_, *h_b_, kBytesB);

    // allocate device memory for output data and initialize to zero
    d_c_ = std::make_unique<cu::DeviceMemory>(kBytesC);
    d_c_->zero(kBytesC);
  }

  void verify_output() {
    // copy C to host
    stream_->memcpyDtoHAsync(*h_c_, *d_c_, kBytesC);
    stream_->synchronize();

    // verify output
    verify<Tin, Tout, kBatchSize, kGlobalM, kGlobalN, kGlobalK>(
        static_cast<const Tin *>(*h_a_), static_cast<const Tin *>(*h_b_),
        static_cast<Tout *>(*h_c_));
  }

protected:
  void complex_gemm_basic() {
    initialize_memory();

    ccglib::mma::GEMM gemm_mma(kBatchSize, kGlobalM, kGlobalK, kGlobalN,
                               kNrInputBits, kNrOutputBits, *device_, *stream_,
                               ccglib::mma::GEMM::basic);

    gemm_mma.run(*d_a_, *d_b_, *d_c_);

    verify_output();
  }

  void complex_gemm_opt() {
    initialize_memory();

    // Allocate device memory for transposed input data
    cu::DeviceMemory d_a_trans(kBytesA);
    cu::DeviceMemory d_b_trans(kBytesB);

    // Transpose A
    ccglib::transpose::Transpose transpose_a(kBatchSize, kGlobalM, kGlobalK,
                                             kMPerBlock, kKPerWMMA,
                                             kNrInputBits, *device_, *stream_);
    transpose_a.run(*h_a_, d_a_trans);

    // Transpose B
    ccglib::transpose::Transpose transpose_b(kBatchSize, kGlobalN, kGlobalK,
                                             kNPerBlock, kKPerWMMA,
                                             kNrInputBits, *device_, *stream_);
    transpose_b.run(*h_b_, d_b_trans);

    ccglib::mma::GEMM gemm_mma(kBatchSize, kGlobalM, kGlobalK, kGlobalN,
                               kNrInputBits, kNrOutputBits, *device_, *stream_,
                               ccglib::mma::GEMM::opt);

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
