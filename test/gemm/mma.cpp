#include <complex>
#include <iostream>
#include <type_traits>

#include <limits.h>
#include <math.h>

#include <cudawrappers/cu.hpp>
#include <cudawrappers/nvrtc.hpp>

#include <ccglib/fp16.h>
#include <ccglib/gemm/mma.h>
#include <ccglib/gemm/reference.h>
#include <ccglib/helper.h>
#include <ccglib/transpose/transpose.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "verify.h"

#ifndef COMPLEX
#define COMPLEX 2
#endif

static inline float float_to_tf32(float x) {
  // converting float to tf32 is equivalent to zeroing the last 13 bits
  float value = x;
  int *value_int = reinterpret_cast<int *>(&value);
  *value_int &= 0xffffe000; // 32 bits: 19 ones followed by 13 zeroes
  return *reinterpret_cast<float *>(value_int);
}

namespace ccglib::test {

template <typename Tin, typename Tout, size_t NrInputBits,
          ccglib::mma::Precision Precision>
class ComplexGemmTestFixture {
public:
  ComplexGemmTestFixture() {
    cu::init();
    device_ = std::make_unique<cu::Device>(0);
    context_ =
        std::make_unique<cu::Context>(CU_CTX_SCHED_BLOCKING_SYNC, *device_);
    stream_ = std::make_unique<cu::Stream>();
  }

  void init(size_t m, size_t n, size_t k) {
    const dim3 dimensions =
        ccglib::mma::GEMM::GetDimensions(Precision, ccglib::mma::opt);
    m_per_block_ = dimensions.x;
    n_per_block_ = dimensions.y;
    k_per_wmma_ = dimensions.z;

    global_m_ = m;
    global_n_ = n;
    global_k_ = k;

    const size_t global_m_padded_ =
        helper::ceildiv(m, m_per_block_) * m_per_block_;
    const size_t global_n_padded_ =
        helper::ceildiv(n, n_per_block_) * n_per_block_;
    const size_t global_k_padded_ =
        helper::ceildiv(k, k_per_wmma_) * k_per_wmma_;

    const size_t kPackingFactor = sizeof(Tin) * CHAR_BIT / NrInputBits;
    bytes_a_ = sizeof(Tin) * kBatchSize * COMPLEX * global_m_ * global_k_ /
               kPackingFactor;
    bytes_b_ = sizeof(Tin) * kBatchSize * COMPLEX * global_n_ * global_k_ /
               kPackingFactor;
    bytes_c_ = sizeof(Tout) * kBatchSize * COMPLEX * global_m_ * global_n_;

    bytes_a_padded_ = sizeof(Tin) * kBatchSize * COMPLEX * global_m_padded_ *
                      global_k_padded_ / kPackingFactor;
    bytes_b_padded_ = sizeof(Tin) * kBatchSize * COMPLEX * global_n_padded_ *
                      global_k_padded_ / kPackingFactor;
    bytes_c_padded_ = sizeof(Tout) * kBatchSize * COMPLEX * global_m_padded_ *
                      global_n_padded_;
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

  // data size and type
  size_t m_per_block_;
  size_t n_per_block_;
  size_t k_per_wmma_;

  size_t global_m_;
  size_t global_n_;
  size_t global_k_;
  const size_t kBatchSize = 16;

  size_t bytes_a_;
  size_t bytes_b_;
  size_t bytes_c_;
  size_t bytes_a_padded_;
  size_t bytes_b_padded_;
  size_t bytes_c_padded_;

  template <typename T> void init_input_matrices(T *a, T *b) {
    // fill a and b with random values (fixed seed), initalize c to zero
    if constexpr (std::is_same_v<T, __half>) {
      unsigned int seed = 0;
      for (int idx = 0; idx < bytes_a_ / sizeof(T); idx++) {
        a[idx] = __float2half(static_cast<float>(rand_r(&seed)) /
                              static_cast<float>(RAND_MAX));
      }
      for (int idx = 0; idx < bytes_b_ / sizeof(T); idx++) {
        b[idx] = __float2half(static_cast<float>(rand_r(&seed)) /
                              static_cast<float>(RAND_MAX));
      }
    } else if constexpr (std::is_same_v<T, float>) {
      unsigned int seed = 0;
      for (int idx = 0; idx < bytes_a_ / sizeof(T); idx++) {
        a[idx] =
            static_cast<float>(rand_r(&seed)) / static_cast<float>(RAND_MAX);
      }
      for (int idx = 0; idx < bytes_b_ / sizeof(T); idx++) {
        b[idx] =
            static_cast<float>(rand_r(&seed)) / static_cast<float>(RAND_MAX);
      }
      // AMD uses actual FP32, NVIDIA uses TF32
#if !defined(__HIP_PLATFORM_AMD__)
      for (int idx = 0; idx < bytes_a_ / sizeof(T); idx++) {
        a[idx] = float_to_tf32(a[idx]);
      }
      for (int idx = 0; idx < bytes_b_ / sizeof(T); idx++) {
        b[idx] = float_to_tf32(b[idx]);
      }
#endif
    } else if constexpr (std::is_same_v<T, unsigned int>) {
      unsigned int seed = 0;
      for (int idx = 0; idx < bytes_a_ / sizeof(T); idx++) {
        a[idx] = static_cast<unsigned int>(rand_r(&seed));
      }
      for (int idx = 0; idx < bytes_b_ / sizeof(T); idx++) {
        b[idx] = static_cast<unsigned int>(rand_r(&seed));
      }
    } else {
      throw std::runtime_error(
          "No init input matrices implemented for this type");
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

  void verify_output(ccglib::mma::MemOrder output_mem_order) {
    // copy C to host
    stream_->memcpyDtoHAsync(*h_c_, *d_c_, bytes_c_);
    stream_->synchronize();

    // verify output
    verify<Tin, Tout, NrInputBits>(
        static_cast<const Tin *>(*h_a_), static_cast<const Tin *>(*h_b_),
        static_cast<Tout *>(*h_c_), kBatchSize, global_m_, global_n_, global_k_,
        output_mem_order);
  }

protected:
  void complex_gemm_basic(ccglib::mma::MemOrder output_mem_order) {
    initialize_memory();

    ccglib::mma::GEMM gemm_mma(kBatchSize, global_m_, global_n_, global_k_,
                               NrInputBits, *device_, *stream_, Precision,
                               ccglib::mma::basic, output_mem_order);
    gemm_mma.Run(*d_a_, *d_b_, *d_c_);

    verify_output(output_mem_order);
  }

  void complex_gemm_opt(ccglib::mma::MemOrder output_mem_order) {
    initialize_memory();

    // Allocate device memory for transposed input data
    cu::DeviceMemory d_a_trans(bytes_a_padded_);
    cu::DeviceMemory d_b_trans(bytes_b_padded_);

    // Transpose A
    ccglib::transpose::Transpose transpose_a(kBatchSize, global_m_, global_k_,
                                             m_per_block_, k_per_wmma_,
                                             NrInputBits, *device_, *stream_);
    transpose_a.Run(*h_a_, d_a_trans);

    // Transpose B
    ccglib::transpose::Transpose transpose_b(kBatchSize, global_n_, global_k_,
                                             n_per_block_, k_per_wmma_,
                                             NrInputBits, *device_, *stream_);
    transpose_b.Run(*h_b_, d_b_trans);

    ccglib::mma::GEMM gemm_mma(kBatchSize, global_m_, global_n_, global_k_,
                               NrInputBits, *device_, *stream_, Precision,
                               ccglib::mma::opt, output_mem_order);

    gemm_mma.Run(d_a_trans, d_b_trans, *d_c_);

    verify_output(output_mem_order);
  }
};

using ComplexGemmTestFixtureFloat16 =
    ComplexGemmTestFixture<half, float, 16, ccglib::mma::float16>;
// on AMD, FP32 is only available on CDNA-class GPUs (GFX9)
// on NVIDIA, TF32 is used
#if !defined(__HIP_PLATFORM_AMD__) ||                                          \
    (defined(__HIP_PLATFORM_AMD__) && defined __GFX9__)
using ComplexGemmTestFixtureFloat32 =
    ComplexGemmTestFixture<float, float, 32, ccglib::mma::float32>;
#endif
// int1 is only available on NVIDIA
#if !defined(__HIP_PLATFORM_AMD__)
using ComplexGemmTestFixtureInt1 =
    ComplexGemmTestFixture<unsigned int, int32_t, 1, ccglib::mma::int1>;
#endif

TEST_CASE_METHOD(ComplexGemmTestFixtureFloat16,
                 "Complex GEMM Test - float16 basic",
                 "[complex-gemm-test-float16-basic]") {
  SECTION("C row-major") {
    const size_t M = 100;
    const size_t N = 60;
    const size_t K = 40;
    ComplexGemmTestFixtureFloat16::init(M, N, K);
    ComplexGemmTestFixtureFloat16::complex_gemm_basic(ccglib::mma::row_major);
  }
  SECTION("C col-major") {
    const size_t M = 128;
    const size_t N = 64;
    const size_t K = 64;
    ComplexGemmTestFixtureFloat16::init(M, N, K);
    ComplexGemmTestFixtureFloat16::complex_gemm_basic(ccglib::mma::col_major);
  }
}

TEST_CASE_METHOD(ComplexGemmTestFixtureFloat16,
                 "Complex GEMM Test - float16 opt",
                 "[complex-gemm-test-float16-opt]") {
  SECTION("C row-major") {
    const size_t M = 100;
    const size_t N = 60;
    const size_t K = 40;
    ComplexGemmTestFixtureFloat16::init(M, N, K);
    ComplexGemmTestFixtureFloat16::complex_gemm_opt(ccglib::mma::row_major);
  }
  SECTION("C col-major") {
    const size_t M = 128;
    const size_t N = 64;
    const size_t K = 64;
    ComplexGemmTestFixtureFloat16::init(M, N, K);
    ComplexGemmTestFixtureFloat16::complex_gemm_opt(ccglib::mma::col_major);
  }
}

#if !defined(__HIP_PLATFORM_AMD__) ||                                          \
    (defined(__HIP_PLATFORM_AMD__) && defined __GFX9__)
TEST_CASE_METHOD(ComplexGemmTestFixtureFloat32,
                 "Complex GEMM Test - float32 basic",
                 "[complex-gemm-test-float32-basic]") {
  SECTION("C row-major") {
    const size_t M = 100;
    const size_t N = 60;
    const size_t K = 40;
    ComplexGemmTestFixtureFloat32::init(M, N, K);
    ComplexGemmTestFixtureFloat32::complex_gemm_basic(ccglib::mma::row_major);
  }
  SECTION("C col-major") {
    const size_t M = 128;
    const size_t N = 64;
    const size_t K = 64;
    ComplexGemmTestFixtureFloat32::init(M, N, K);
    ComplexGemmTestFixtureFloat32::complex_gemm_basic(ccglib::mma::col_major);
  }
}

TEST_CASE_METHOD(ComplexGemmTestFixtureFloat32,
                 "Complex GEMM Test - float32 opt",
                 "[complex-gemm-test-float32-opt]") {
  SECTION("C row-major") {
    const size_t M = 100;
    const size_t N = 60;
    const size_t K = 40;
    ComplexGemmTestFixtureFloat32::init(M, N, K);
    ComplexGemmTestFixtureFloat32::complex_gemm_opt(ccglib::mma::row_major);
  }
  SECTION("C col-major") {
    const size_t M = 128;
    const size_t N = 64;
    const size_t K = 64;
    ComplexGemmTestFixtureFloat32::init(M, N, K);
    ComplexGemmTestFixtureFloat32::complex_gemm_opt(ccglib::mma::col_major);
  }
}
#endif

#if !defined(__HIP_PLATFORM_AMD__)
TEST_CASE_METHOD(ComplexGemmTestFixtureInt1, "Complex GEMM Test - int1 basic",
                 "[complex-gemm-test-int1-basic]") {
  SECTION("C row-major") {
    const size_t M = 64;
    const size_t N = 64;
    const size_t K = 256;
    ComplexGemmTestFixtureInt1::init(M, N, K);
    ComplexGemmTestFixtureInt1::complex_gemm_basic(ccglib::mma::row_major);
  }
  SECTION("C col-major") {
    const size_t M = 64;
    const size_t N = 64;
    const size_t K = 256;
    ComplexGemmTestFixtureInt1::init(M, N, K);
    ComplexGemmTestFixtureInt1::complex_gemm_basic(ccglib::mma::col_major);
  }
}

TEST_CASE_METHOD(ComplexGemmTestFixtureInt1, "Complex GEMM Test - int1 opt",
                 "[complex-gemm-test-int1-opt]") {
  SECTION("C row-major") {
    const size_t M = 64;
    const size_t N = 64;
    const size_t K = 256;
    ComplexGemmTestFixtureInt1::init(M, N, K);
    ComplexGemmTestFixtureInt1::complex_gemm_opt(ccglib::mma::row_major);
  }
  SECTION("C col-major") {
    const size_t M = 64;
    const size_t N = 64;
    const size_t K = 256;
    ComplexGemmTestFixtureInt1::init(M, N, K);
    ComplexGemmTestFixtureInt1::complex_gemm_opt(ccglib::mma::col_major);
  }
}
#endif

TEST_CASE("Unsupported matrix layout") {
  const size_t batch_size = 16;
  const size_t m = 16;
  const size_t n = 16;
  const size_t k = 256;

#if defined(__HIP__)
  const std::string error_name = "HIPRTC_ERROR_COMPILATION";
#else
  const std::string error_name = "NVRTC_ERROR_COMPILATION";
#endif

  cu::init();
  cu::Device device(0);
  cu::Context context(CU_CTX_BLOCKING_SYNC, device);
  cu::Stream stream;

  ccglib::mma::MemOrder layout_a = ccglib::mma::col_major;
  ccglib::mma::MemOrder layout_b = ccglib::mma::col_major;
  ccglib::mma::MemOrder layout_c = ccglib::mma::row_major;

  // float16 could support different layouts, but not currently implemented
  // A must be row-major, B col-major
  SECTION("float16") {
    CHECK_THROWS_WITH(ccglib::mma::GEMM(batch_size, m, n, k, 16, device, stream,
                                        ccglib::mma::float16,
                                        ccglib::mma::basic, layout_c, layout_a,
                                        layout_b),
                      Catch::Matchers::ContainsSubstring(error_name));
  }

#if !defined(__HIP_PLATFORM_AMD__)
  // 1-bit requires A row-major, B col-major
  SECTION("int1") {
    CHECK_THROWS_WITH(ccglib::mma::GEMM(batch_size, m, n, k, 1, device, stream,
                                        ccglib::mma::int1, ccglib::mma::basic,
                                        layout_c, layout_a, layout_b),
                      Catch::Matchers::ContainsSubstring(error_name));
  }
#endif
}

} // namespace ccglib::test
