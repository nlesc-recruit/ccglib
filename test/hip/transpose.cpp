#include <catch2/catch_test_macros.hpp>
#include <hip/hip_runtime.h>
#include <iostream>
#include <limits>

#include <xtensor/xadapt.hpp>
#include <xtensor/xtensor.hpp>

#include <ccglib/fp16.h>
#include <ccglib/precision.h>
#include <ccglib/transpose/transpose.h>

namespace ccglib::test {

static inline void hip_check(hipError_t err) {
  if (err != hipSuccess) {
    throw std::runtime_error(hipGetErrorString(err));
  }
}

template <typename T, ccglib::ValueType InputPrecision>
class TransposeTestFixture {
public:
  TransposeTestFixture() {
    hip_check(hipInit(0));
    hipDevice_t device;
    hip_check(hipDeviceGet(&device, 0));
    hipStream_t stream;
    hip_check(hipStreamCreate(&stream));
  }

  ~TransposeTestFixture() {
    hip_check(hipHostFree(h_a_));
    hip_check(hipHostFree(h_a_trans_));
    hip_check(hipFree(d_a_));
    hip_check(hipFree(d_a_trans_));
  }

private:
  hipDevice_t device_;
  hipStream_t stream_;

  T *h_a_ = nullptr;
  T *h_a_trans_ = nullptr;
  hipDeviceptr_t d_a_ = 0;
  hipDeviceptr_t d_a_trans_ = 0;

  const size_t kComplex = 2;
  const size_t kBatchSize = 2;
  const size_t kGlobalM = 256; // must be a multiple of kMPerChunk
  const size_t kGlobalN = 128; // must be a multiple of kNPerChunk
  const size_t kMPerChunk = 64;
  const size_t kNPerChunk = 64; // must be a multiple of kPackingFactor

  const size_t kPackingFactor =
      sizeof(T) * CHAR_BIT /
      ccglib::ValuePrecision{InputPrecision}.GetBitWidth();
  const size_t kBytesA =
      sizeof(T) * kBatchSize * kComplex * kGlobalM * kGlobalN / kPackingFactor;

  void init_input_matrix(T *a) {
    unsigned int seed = 0;
    if constexpr (std::is_same_v<T, __half>) {
      const float scale = 1.0f;
      for (size_t idx = 0; idx < kBytesA / sizeof(T); ++idx) {
        a[idx] = __float2half(2.0f * scale *
                                  (static_cast<float>(rand_r(&seed)) /
                                   static_cast<float>(RAND_MAX)) -
                              scale);
      }
    } else if constexpr (std::is_same_v<T, unsigned int>) {
      for (size_t idx = 0; idx < kBytesA / sizeof(T); ++idx) {
        a[idx] = static_cast<unsigned int>(rand_r(&seed));
      }
    }
  }

  void init_memory() {
    // initialize host memory
    hip_check(hipHostMalloc(&h_a_, kBytesA));
    hip_check(hipHostMalloc(&h_a_trans_, kBytesA));
    init_input_matrix(h_a_);

    // Allocate device memory for input data
    hip_check(hipMalloc(&d_a_, kBytesA));

    // Transfer the input data
    hip_check(hipMemcpyHtoDAsync(d_a_, h_a_, kBytesA, stream_));

    // allocate device memory for output data and initialize to zero
    hip_check(hipMalloc(&d_a_trans_, kBytesA));
    hip_check(hipMemsetD8Async(d_a_trans_, 0, kBytesA, stream_));
  }

  void verify_output(transpose::ComplexAxisLocation complex_axis_location) {
    // copy output to host
    hip_check(hipMemcpyDtoHAsync(h_a_trans_, d_a_trans_, kBytesA, stream_));
    hip_check(hipStreamSynchronize(stream_));

    // verification
    // data is transposed from shape [batch][complex][m][n / packing_factor] to
    // [batch][m/m_per_chunk][n/n_per_chunk][complex][m_per_chunk][n_per_chunk /
    // packing_factor] in complex-middle mode, where packing_factor is the
    // number of samples per item (e.g. 32 for 1-bit samples packed into 32-bit
    // ints.) In complex-last mode, complex is the last axis.

    std::array<size_t, 4> shape_input;
    if (complex_axis_location ==
        transpose::ComplexAxisLocation::complex_middle) {
      shape_input = {kBatchSize, kComplex, kGlobalM, kGlobalN / kPackingFactor};
    } else {
      shape_input = {kBatchSize, kGlobalM, kGlobalN / kPackingFactor, kComplex};
    }

    std::array<size_t, 6> shape_output = {
        kBatchSize,
        kGlobalM / kMPerChunk,
        kGlobalN / kNPerChunk,
        kComplex,
        kMPerChunk,
        kNPerChunk / kPackingFactor,
    };

    auto input = xt::adapt(reinterpret_cast<T *>(h_a_), shape_input);
    auto output = xt::adapt(reinterpret_cast<T *>(h_a_trans_), shape_output);

    for (size_t b = 0; b < kBatchSize; b++) {
      for (size_t c = 0; c < kComplex; c++) {
        for (size_t m = 0; m < kGlobalM; m++) {
          const size_t m_local = m / kMPerChunk;
          const size_t m_chunk = m % kMPerChunk;

          for (size_t n = 0; n < kGlobalN / kPackingFactor; n++) {
            const size_t n_local = n / (kNPerChunk / kPackingFactor);
            const size_t n_chunk = n % (kNPerChunk / kPackingFactor);

            float in;
            if (complex_axis_location ==
                transpose::ComplexAxisLocation::complex_middle) {
              in = static_cast<float>(input(b, c, m, n));
            } else {
              in = static_cast<float>(input(b, m, n, c));
            }

            const float out = static_cast<float>(
                output(b, m_local, n_local, c, m_chunk, n_chunk));

            REQUIRE(in == out);
          }
        }
      }
    }
  }

public:
  void transpose(transpose::ComplexAxisLocation complex_axis_location) {
    init_memory();

    ccglib::transpose::Transpose transpose_a(
        kBatchSize, kGlobalM, kGlobalN, kMPerChunk, kNPerChunk,
        ccglib::ValuePrecision(InputPrecision).GetBitWidth(), device_, stream_,
        complex_axis_location);

    transpose_a.Run(d_a_, d_a_trans_);

    verify_output(complex_axis_location);
  }
};

using TransposeTestFixtureFloat16 =
    TransposeTestFixture<half, ccglib::ValueType::float16>;
using TransposeTestFixtureInt1 =
    TransposeTestFixture<unsigned int, ccglib::ValueType::int1>;

TEST_CASE_METHOD(TransposeTestFixtureFloat16, "Transpose Test - float16",
                 "[transpose-test-float16]") {
  SECTION("complex-middle") {
    transpose(transpose::ComplexAxisLocation::complex_middle);
  }
  SECTION("complex-last") {
    transpose(transpose::ComplexAxisLocation::complex_last);
  }
}

TEST_CASE_METHOD(TransposeTestFixtureInt1, "Transpose Test - int1",
                 "[transpose-test-int1]") {
  transpose(transpose::ComplexAxisLocation::complex_middle);
}

} // namespace ccglib::test
