#include <complex>
#include <limits.h>
#include <math.h>

#include <cudawrappers/cu.hpp>
#include <cudawrappers/nvrtc.hpp>

#include <ccglib/bf16.h>
#include <ccglib/common/arch.h>
#include <ccglib/common/helper.h>
#include <ccglib/common/precision.h>
#include <ccglib/fp16.h>
#include <ccglib/gemm/mma.h>
#include <ccglib/gemm/reference.h>
#include <ccglib/transpose/transpose.h>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <xtensor/xadapt.hpp>
#include <xtensor/xtensor.hpp>

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

static inline std::string type_to_string(ccglib::ValueType type) {
  switch (type) {
  case ccglib::int1:
    return "int1";
  case ccglib::int32:
    return "int32";
  case ccglib::bfloat16:
    return "bfloat16";
  case ccglib::float16:
    return "float16";
  case ccglib::float32:
    return "float32";
  default:
    return "unrecognized type";
  }
}

namespace ccglib::test {

template <typename Tin, typename Tout, ccglib::ValueType InputPrecision,
          ccglib::ValueType OutputPrecision>
class ComplexGemmTestFixture {
public:
  using InputType = Tin;
  using OutputType = Tout;
  const std::string InputTypeName = type_to_string(InputPrecision);
  const std::string OutputTypeName = type_to_string(OutputPrecision);
  std::unique_ptr<cu::Device> device_;

  ComplexGemmTestFixture() {
    cu::init();
    device_ = std::make_unique<cu::Device>(0);
    context_ =
        std::make_unique<cu::Context>(CU_CTX_SCHED_BLOCKING_SYNC, *device_);
    stream_ = std::make_unique<cu::Stream>();
  }

  void init(size_t m, size_t n, size_t k) {
    const dim3 dimensions = ccglib::mma::GEMM::GetDimensions(
        {InputPrecision, OutputPrecision}, ccglib::mma::opt);
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

    const size_t kPackingFactor =
        sizeof(Tin) * CHAR_BIT /
        ccglib::ValuePrecision{InputPrecision}.GetBitWidth();
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
    } else if constexpr (std::is_same_v<T, bf16>) {
      unsigned int seed = 0;
      for (int idx = 0; idx < bytes_a_ / sizeof(T); idx++) {
        a[idx] = static_cast<bf16>(static_cast<float>(rand_r(&seed)) /
                                   static_cast<float>(RAND_MAX));
      }
      for (int idx = 0; idx < bytes_b_ / sizeof(T); idx++) {
        b[idx] = static_cast<bf16>(static_cast<float>(rand_r(&seed)) /
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
    verify<Tin, Tout, InputPrecision>(
        static_cast<const Tin *>(*h_a_), static_cast<const Tin *>(*h_b_),
        static_cast<Tout *>(*h_c_), kBatchSize, global_m_, global_n_, global_k_,
        output_mem_order);
  }

public:
  void complex_gemm_basic(ccglib::mma::MemOrder output_mem_order) {
    initialize_memory();

    ccglib::mma::GEMM gemm_mma(
        kBatchSize, global_m_, global_n_, global_k_, *device_, *stream_,
        {InputPrecision, OutputPrecision}, ccglib::mma::basic,
        ccglib::complex_planar, output_mem_order);

    gemm_mma.Run(*d_a_, *d_b_, *d_c_);

    verify_output(output_mem_order);
  }

  void complex_gemm_opt(ccglib::mma::MemOrder output_mem_order) {
    initialize_memory();

    // Allocate device memory for transposed input data
    cu::DeviceMemory d_a_trans(bytes_a_padded_);
    cu::DeviceMemory d_b_trans(bytes_b_padded_);

    // Transpose A
    ccglib::transpose::Transpose transpose_a(
        kBatchSize, global_m_, global_k_, m_per_block_, k_per_wmma_,
        ccglib::ValuePrecision{InputPrecision}.GetBitWidth(), *device_,
        *stream_);
    transpose_a.Run(*h_a_, d_a_trans);

    // Transpose B
    ccglib::transpose::Transpose transpose_b(
        kBatchSize, global_n_, global_k_, n_per_block_, k_per_wmma_,
        ccglib::ValuePrecision{InputPrecision}.GetBitWidth(), *device_,
        *stream_);
    transpose_b.Run(*h_b_, d_b_trans);

    ccglib::mma::GEMM gemm_mma(
        kBatchSize, global_m_, global_n_, global_k_, *device_, *stream_,
        {InputPrecision, OutputPrecision}, ccglib::mma::opt,
        ccglib::complex_planar, output_mem_order);

    gemm_mma.Run(d_a_trans, d_b_trans, *d_c_);

    verify_output(output_mem_order);
  }

  void complex_gemm_opt(ccglib::mma::MemOrder output_mem_order,
                        ccglib::ComplexAxisLocation complex_axis_location) {
    initialize_memory();

    // Allocate device memory for transposed input data
    cu::DeviceMemory d_a_trans(bytes_a_padded_);
    cu::DeviceMemory d_b_trans(bytes_b_padded_);

    // Transpose A
    ccglib::transpose::Transpose transpose_a(
        kBatchSize, global_m_, global_k_, m_per_block_, k_per_wmma_,
        ccglib::ValuePrecision{InputPrecision}.GetBitWidth(), *device_,
        *stream_);
    transpose_a.Run(*h_a_, d_a_trans);

    // Transpose B
    ccglib::transpose::Transpose transpose_b(
        kBatchSize, global_n_, global_k_, n_per_block_, k_per_wmma_,
        ccglib::ValuePrecision{InputPrecision}.GetBitWidth(), *device_,
        *stream_);
    transpose_b.Run(*h_b_, d_b_trans);

    ccglib::mma::GEMM gemm_mma(
        kBatchSize, global_m_, global_n_, global_k_, *device_, *stream_,
        {InputPrecision, OutputPrecision}, ccglib::mma::opt,
        complex_axis_location, output_mem_order);

    gemm_mma.Run(d_a_trans, d_b_trans, *d_c_);

    // convert the output from complex-interleaved to to complex-planar layout
    // and reuse the verify function that expects complex-planar layout
    if (complex_axis_location == ccglib::complex_interleaved) {
      // copy C to host
      stream_->memcpyDtoHAsync(*h_c_, *d_c_, bytes_c_);
      stream_->synchronize();

      // move complex axis to planar
      const std::array<size_t, 3> shape{kBatchSize, global_m_ * global_n_,
                                        COMPLEX};
      auto h_c_complex_interleaved =
          xt::adapt(static_cast<Tout *>(*h_c_), shape);
      xt::xtensor<Tout, 3> h_c_complex_planar =
          xt::transpose(h_c_complex_interleaved, {0, 2, 1});

      // verify output
      verify<Tin, Tout, InputPrecision>(
          static_cast<const Tin *>(*h_a_), static_cast<const Tin *>(*h_b_),
          static_cast<Tout *>(h_c_complex_planar.data()), kBatchSize, global_m_,
          global_n_, global_k_, output_mem_order);
    } else {
      verify_output(output_mem_order);
    }
  }
};

using TestTypesComplexGemm = std::tuple<
// CUDA does not support bfloat16 as accumulation type
#ifdef __HIP_PLATFORM_AMD__
    ComplexGemmTestFixture<bf16, bf16, ccglib::ValueType::bfloat16,
                           ccglib::ValueType::bfloat16>,
#endif
    ComplexGemmTestFixture<float, bf16, ccglib::ValueType::float32,
                           ccglib::ValueType::bfloat16>,
    ComplexGemmTestFixture<bf16, float, ccglib::ValueType::bfloat16,
                           ccglib::ValueType::float32>,
    ComplexGemmTestFixture<half, half, ccglib::ValueType::float16,
                           ccglib::ValueType::float16>,
    ComplexGemmTestFixture<float, half, ccglib::ValueType::float32,
                           ccglib::ValueType::float16>,
    ComplexGemmTestFixture<half, float, ccglib::ValueType::float16,
                           ccglib::ValueType::float32>,
    ComplexGemmTestFixture<float, float, ccglib::ValueType::float32,
                           ccglib::ValueType::float32>>;

// GemmTestTraits is a template struct that provides parameters for the
// GemmTestBasic and GemmTestOpt test cases.
//
// It is specialized for each Fixture type in the TestTypesComplexGemm tuple.
//
// In the absence of a specialization, it will trigger a static assertion.
template <typename Fixture, typename Cond = void> struct GemmTestTraits {
  static_assert(!std::is_void_v<Cond>,
                "GemmTestTraits not specialized for this Fixture type.");
};

template <typename T, typename Tuple> struct IsTypeInTuple;

template <typename T, typename... Types>
struct IsTypeInTuple<T, std::tuple<Types...>>
    : std::disjunction<std::is_same<T, Types>...> {};

template <typename T, typename Tuple>
constexpr bool IsTypeInTuple_v = IsTypeInTuple<T, Tuple>::value;

struct TestDimensions {
  size_t aligned;
  size_t unaligned;
};

template <typename Fixture>
struct GemmTestTraits<
    Fixture, std::enable_if_t<IsTypeInTuple_v<Fixture, TestTypesComplexGemm>>> {
  static constexpr TestDimensions M_row_major = {128, 100};
  static constexpr TestDimensions N_row_major = {256, 60};
  static constexpr TestDimensions K_row_major = {64, 40};

  static constexpr TestDimensions M_col_major = {128, 100};
  static constexpr TestDimensions N_col_major = {256, 60};
  static constexpr TestDimensions K_col_major = {64, 40};

  static constexpr size_t InputSize =
      sizeof(typename Fixture::InputType) * CHAR_BIT;
  static constexpr size_t OutputSize =
      sizeof(typename Fixture::OutputType) * CHAR_BIT;
};

// int1 is only available on NVIDIA
#if !defined(__HIP_PLATFORM_AMD__)
using ComplexGemmTestFixtureInt1 =
    ComplexGemmTestFixture<unsigned int, int32_t, ccglib::ValueType::int1,
                           ccglib::ValueType::int32>;

template <typename Fixture>
struct GemmTestTraits<
    Fixture,
    std::enable_if_t<std::is_same_v<Fixture, ComplexGemmTestFixtureInt1>>> {
  // Unaligned GEMM is not supported for int1, therefore the
  // parameters are kept the same as for the aligned case
  static constexpr TestDimensions M_row_major = {64, 64};
  static constexpr TestDimensions N_row_major = {64, 64};
  static constexpr TestDimensions K_row_major = {256, 256};

  static constexpr TestDimensions M_col_major = {64, 64};
  static constexpr TestDimensions N_col_major = {64, 64};
  static constexpr TestDimensions K_col_major = {256, 256};

  static constexpr size_t InputSize = 1;
  static constexpr size_t OutputSize =
      sizeof(typename Fixture::OutputType) * CHAR_BIT;
};
#endif

template <typename Fixture> struct GemmTestBasic : public Fixture {
  void run_tests() {
    using Traits = GemmTestTraits<Fixture>;

    SECTION("basic-row-major - InputType: " + this->InputTypeName +
            ", OutputType: " + this->OutputTypeName) {
      this->init(Traits::M_row_major.aligned, Traits::N_row_major.aligned,
                 Traits::K_row_major.aligned);
      this->complex_gemm_basic(ccglib::mma::row_major);

      this->init(Traits::M_row_major.unaligned, Traits::N_row_major.unaligned,
                 Traits::K_row_major.unaligned);
      this->complex_gemm_basic(ccglib::mma::row_major);
    }
    SECTION("basic-col-major - InputType: " + this->InputTypeName +
            ", OutputType: " + this->OutputTypeName) {
      this->init(Traits::M_col_major.aligned, Traits::N_col_major.aligned,
                 Traits::K_col_major.aligned);
      this->complex_gemm_basic(ccglib::mma::col_major);

      this->init(Traits::M_row_major.unaligned, Traits::N_row_major.unaligned,
                 Traits::K_row_major.unaligned);
      this->complex_gemm_basic(ccglib::mma::row_major);
    }
  }
};

template <typename Fixture> struct GemmTestOpt : public Fixture {
  void run_tests() {
    using Traits = GemmTestTraits<Fixture>;

    SECTION("opt-row-major - InputType: " + this->InputTypeName +
            ", OutputType: " + this->OutputTypeName) {
      this->init(Traits::M_row_major.aligned, Traits::N_row_major.aligned,
                 Traits::K_row_major.aligned);
      this->complex_gemm_opt(ccglib::mma::row_major);

      this->init(Traits::M_row_major.unaligned, Traits::N_row_major.unaligned,
                 Traits::K_row_major.unaligned);
      this->complex_gemm_basic(ccglib::mma::row_major);
    }
    SECTION("opt-col-major - InputType: " + this->InputTypeName +
            ", OutputType: " + this->OutputTypeName) {
      this->init(Traits::M_col_major.aligned, Traits::N_col_major.aligned,
                 Traits::K_col_major.aligned);
      this->complex_gemm_opt(ccglib::mma::col_major);

      this->init(Traits::M_col_major.unaligned, Traits::N_col_major.unaligned,
                 Traits::K_col_major.unaligned);
      this->complex_gemm_opt(ccglib::mma::col_major);
    }

    SECTION("opt-row-major-complex-interleaved - InputType: " +
            this->InputTypeName + ", OutputType: " + this->OutputTypeName) {
      this->init(Traits::M_row_major.aligned, Traits::N_row_major.aligned,
                 Traits::K_row_major.aligned);
      this->complex_gemm_opt(ccglib::mma::row_major,
                             ccglib::complex_interleaved);

      this->init(Traits::M_row_major.unaligned, Traits::N_row_major.unaligned,
                 Traits::K_row_major.unaligned);
      this->complex_gemm_opt(ccglib::mma::row_major,
                             ccglib::complex_interleaved);
    }

    SECTION("opt-col-major-complex-interleaved - InputType: " +
            this->InputTypeName + ", OutputType: " + this->OutputTypeName) {
      this->init(Traits::M_col_major.aligned, Traits::N_col_major.aligned,
                 Traits::K_col_major.aligned);
      this->complex_gemm_opt(ccglib::mma::col_major,
                             ccglib::complex_interleaved);

      this->init(Traits::M_col_major.unaligned, Traits::N_col_major.unaligned,
                 Traits::K_col_major.unaligned);
      this->complex_gemm_opt(ccglib::mma::col_major,
                             ccglib::complex_interleaved);
    }
  }
};

TEMPLATE_LIST_TEST_CASE_METHOD(GemmTestBasic, "Complex GEMM Test",
                               "[complex-gemm-test-basic]",
                               TestTypesComplexGemm) {
  if constexpr (std::is_same_v<typename GemmTestBasic<TestType>::InputType,
                               float>) {

    // on AMD, skip on unsupported GPUs
#ifdef __HIP_PLATFORM_AMD__
    if (!isCDNA(*GemmTestBasic<TestType>().device_)) {
      SKIP("Float32 is only available on CDNA GPUs");
    }
// on NVIDIA, skip on unsupported GPUs
#else
    if (isVolta(*GemmTestBasic<TestType>().device_)) {
      SKIP("Float32 is not available on Volta GPUs");
    }
#endif
  }

  GemmTestBasic<TestType>().run_tests();
}

TEMPLATE_LIST_TEST_CASE_METHOD(GemmTestOpt, "Complex GEMM Test",
                               "[complex-gemm-test-opt]",
                               TestTypesComplexGemm) {
  // on AMD, skip on unsupported GPUs
#ifdef __HIP_PLATFORM_AMD__
  if (!isCDNA(*GemmTestBasic<TestType>().device_)) {
    SKIP("Float32 is only available on CDNA GPUs");
  }
// on NVIDIA, skip on unsupported GPUs
#else
  if (isVolta(*GemmTestBasic<TestType>().device_)) {
    SKIP("Float32 is not available on Volta GPUs");
  }
#endif

  GemmTestOpt<TestType>().run_tests();
}

#if !defined(__HIP_PLATFORM_AMD__)
TEST_CASE_METHOD(ComplexGemmTestFixtureInt1, "Complex GEMM Test - int1 basic",
                 "[complex-gemm-test-int1-basic]") {
  if (isVolta(*ComplexGemmTestFixtureInt1().device_)) {
    SKIP("Int1 is not available on Volta GPUs");
  }
  GemmTestBasic<ComplexGemmTestFixtureInt1>().run_tests();
}

TEST_CASE_METHOD(ComplexGemmTestFixtureInt1, "Complex GEMM Test - int1 opt",
                 "[complex-gemm-test-int1-opt]") {
  if (isVolta(*ComplexGemmTestFixture().device_)) {
    SKIP("Int1 is not available on Volta GPUs");
  }
  GemmTestOpt<ComplexGemmTestFixtureInt1>().run_tests();
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
    CHECK_THROWS_WITH(
        ccglib::mma::GEMM(batch_size, m, n, k, device, stream,
                          ccglib::ValueType::float16, ccglib::mma::basic,
                          ccglib::complex_planar, layout_c, layout_a, layout_b),
        Catch::Matchers::ContainsSubstring(error_name));
  }

#if !defined(__HIP_PLATFORM_AMD__)
  // 1-bit requires A row-major, B col-major
  SECTION("int1") {
    if (isVolta(device)) {
      SKIP("Int1 is not available on Volta GPUs");
    }
    CHECK_THROWS_WITH(
        ccglib::mma::GEMM(batch_size, m, n, k, device, stream,
                          {ccglib::ValueType::int1, ccglib::ValueType::int32},
                          ccglib::mma::basic, ccglib::complex_planar, layout_c,
                          layout_a, layout_b),
        Catch::Matchers::ContainsSubstring(error_name));
  }
#endif
}

TEST_CASE("Alpha/beta scaling") {
  const size_t batch_size = 16;
  const size_t m = GENERATE(255, 256); // aligned and padded case
  const size_t n = 128;
  const size_t k = 256;

  cu::init();
  cu::Device device(0);
  cu::Context context(CU_CTX_BLOCKING_SYNC, device);
  cu::Stream stream;

  SECTION("float16") {
    using Tin = half;
    using Tout = float;
    const std::complex<float> alpha = {0.3, 1.6};
    const std::complex<float> beta = {0.8, -1.1};

    ccglib::mma::GEMM gemm(
        batch_size, m, n, k, device, stream,
        {ccglib::ValueType::float16, ccglib::ValueType::float32},
        ccglib::mma::basic, ccglib::complex_planar, ccglib::mma::row_major,
        ccglib::mma::row_major, ccglib::mma::col_major, alpha, beta);

    const size_t bytes_a = batch_size * COMPLEX * m * k * sizeof(Tin);
    const size_t bytes_b = batch_size * COMPLEX * n * k * sizeof(Tin);
    const size_t bytes_c = batch_size * COMPLEX * m * n * sizeof(Tout);

    cu::HostMemory h_a(bytes_a);
    cu::HostMemory h_b(bytes_b);
    cu::HostMemory h_c_in(bytes_c);
    cu::HostMemory h_c_out(bytes_c);

    unsigned int seed = 0;
    for (int idx = 0; idx < bytes_a / sizeof(Tin); idx++) {
      static_cast<Tin *>(h_a)[idx] = __float2half(
          static_cast<float>(rand_r(&seed)) / static_cast<float>(RAND_MAX));
    }

    for (int idx = 0; idx < bytes_b / sizeof(Tin); idx++) {
      static_cast<Tin *>(h_b)[idx] = __float2half(
          static_cast<float>(rand_r(&seed)) / static_cast<float>(RAND_MAX));
    }

    for (int idx = 0; idx < bytes_c / sizeof(Tout); idx++) {
      static_cast<Tout *>(h_c_in)[idx] =
          static_cast<float>(rand_r(&seed)) / static_cast<float>(RAND_MAX);
    }

    cu::DeviceMemory d_a(bytes_a);
    cu::DeviceMemory d_b(bytes_b);
    cu::DeviceMemory d_c(bytes_c);

    stream.memcpyHtoDAsync(d_a, h_a, bytes_a);
    stream.memcpyHtoDAsync(d_b, h_b, bytes_b);
    stream.memcpyHtoDAsync(d_c, h_c_in, bytes_c);

    gemm.Run(d_a, d_b, d_c);

    stream.memcpyDtoHAsync(h_c_out, d_c, bytes_c);
    stream.synchronize();

    verify<Tin, Tout, ccglib::ValueType::float16>(
        static_cast<const Tin *>(h_a), static_cast<const Tin *>(h_b),
        static_cast<Tout *>(h_c_out), batch_size, m, n, k,
        ccglib::mma::row_major, alpha, beta, static_cast<Tout *>(h_c_in));
  }

  SECTION("int1") {
#ifdef __HIP_PLATFORM_AMD__
    SKIP("int1 is not available on AMD GPUs");
#endif
    if (isVolta(device)) {
      SKIP("Int1 is not available on Volta GPUs");
    }

    using Tin = unsigned int;
    using Tout = int;
    const std::complex<float> alpha = {2, -3};
    const std::complex<float> beta = {3, -2};

    ccglib::mma::GEMM gemm(batch_size, m, n, k, device, stream,
                           {ccglib::ValueType::int1, ccglib::ValueType::int32},
                           ccglib::mma::basic, ccglib::complex_planar,
                           ccglib::mma::row_major, ccglib::mma::row_major,
                           ccglib::mma::col_major, alpha, beta);

    const size_t bits_per_sample = sizeof(Tin) * CHAR_BIT;
    const size_t k_packed = ccglib::helper::ceildiv(k, bits_per_sample);

    const size_t bytes_a = batch_size * COMPLEX * m * k_packed * sizeof(Tin);
    const size_t bytes_b = batch_size * COMPLEX * n * k_packed * sizeof(Tin);
    const size_t bytes_c = batch_size * COMPLEX * m * n * sizeof(Tout);

    cu::HostMemory h_a(bytes_a);
    cu::HostMemory h_b(bytes_b);
    cu::HostMemory h_c_in(bytes_c);
    cu::HostMemory h_c_out(bytes_c);

    unsigned int seed = 0;
    for (int idx = 0; idx < bytes_a / sizeof(Tin); idx++) {
      static_cast<Tin *>(h_a)[idx] = static_cast<unsigned int>(rand_r(&seed));
    }

    for (int idx = 0; idx < bytes_b / sizeof(Tin); idx++) {
      static_cast<Tin *>(h_b)[idx] = static_cast<unsigned int>(rand_r(&seed));
    }

    for (int idx = 0; idx < bytes_c / sizeof(Tout); idx++) {
      static_cast<Tout *>(h_c_in)[idx] = static_cast<int>(rand_r(&seed));
    }

    cu::DeviceMemory d_a(bytes_a);
    cu::DeviceMemory d_b(bytes_b);
    cu::DeviceMemory d_c(bytes_c);

    stream.memcpyHtoDAsync(d_a, h_a, bytes_a);
    stream.memcpyHtoDAsync(d_b, h_b, bytes_b);
    stream.memcpyHtoDAsync(d_c, h_c_in, bytes_c);

    gemm.Run(d_a, d_b, d_c);

    stream.memcpyDtoHAsync(h_c_out, d_c, bytes_c);
    stream.synchronize();

    verify<Tin, Tout, ccglib::ValueType::int1>(
        static_cast<const Tin *>(h_a), static_cast<const Tin *>(h_b),
        static_cast<Tout *>(h_c_out), batch_size, m, n, k,
        ccglib::mma::row_major, alpha, beta, static_cast<Tout *>(h_c_in));
  }
}

} // namespace ccglib::test
