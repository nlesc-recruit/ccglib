#include <catch2/catch_test_macros.hpp>
#include <cudawrappers/cu.hpp>
#include <iostream>
#include <limits>

#include <xtensor/xadapt.hpp>
#include <xtensor/xtensor.hpp>

#include <ccglib/fp16.h>
#include <ccglib/precision.h>
#include <ccglib/transpose/transpose.h>

namespace ccglib::test {

template <typename T, ccglib::ValueType InputPrecision>
class TransposeTestFixture {
public:
  TransposeTestFixture() {
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
  std::unique_ptr<cu::HostMemory> h_a_trans_;
  std::unique_ptr<cu::DeviceMemory> d_a_;
  std::unique_ptr<cu::DeviceMemory> d_a_trans_;

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
    // fill a (fixed seed)
    if constexpr (std::is_same_v<T, __half>) {
      unsigned int seed = 0;
      const float scale = 1.0f;
      for (int idx = 0; idx < kBytesA / sizeof(T); idx++) {
        a[idx] = __float2half(2.0f * scale *
                                  (static_cast<float>(rand_r(&seed)) /
                                   static_cast<float>(RAND_MAX)) -
                              scale);
      }
    } else if constexpr (std::is_same_v<T, unsigned int>) {
      unsigned int seed = 0;
      for (int idx = 0; idx < kBytesA / sizeof(T); idx++) {
        a[idx] = static_cast<unsigned int>(rand_r(&seed));
      }
    }
  }

  void init_memory() {
    // initalize host memory
    h_a_ = std::make_unique<cu::HostMemory>(kBytesA);
    h_a_trans_ = std::make_unique<cu::HostMemory>(kBytesA);
    init_input_matrix(static_cast<T *>(*h_a_));

    // Allocate device memory for input data
    d_a_ = std::make_unique<cu::DeviceMemory>(kBytesA);

    // Transfer the input data
    stream_->memcpyHtoDAsync(*d_a_, *h_a_, kBytesA);

    // allocate device memory for output data and initialize to zero
    d_a_trans_ = std::make_unique<cu::DeviceMemory>(kBytesA);
    d_a_trans_->zero(kBytesA);
  }

  void verify_output(transpose::ComplexAxisLocation complex_axis_location) {
    // copy output to host
    stream_->memcpyDtoHAsync(*h_a_trans_, *d_a_trans_, kBytesA);
    stream_->synchronize();

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
    } else if (complex_axis_location ==
               transpose::ComplexAxisLocation::complex_last) {
      shape_input = {kBatchSize, kGlobalM, kGlobalN / kPackingFactor, kComplex};
    }

    std::array<size_t, 6> shape_output{
        kBatchSize, kGlobalM / kMPerChunk,      kGlobalN / kNPerChunk, kComplex,
        kMPerChunk, kNPerChunk / kPackingFactor};

    auto input = xt::adapt(static_cast<T *>(*h_a_), shape_input);
    auto output = xt::adapt(static_cast<T *>(*h_a_trans_), shape_output);

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
              in = input(b, c, m, n);
            } else if (complex_axis_location ==
                       transpose::ComplexAxisLocation::complex_last) {
              in = input(b, m, n, c);
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
        ccglib::ValuePrecision(InputPrecision).GetBitWidth(), *device_,
        *stream_, complex_axis_location);
    transpose_a.Run(*d_a_, *d_a_trans_);

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
    TransposeTestFixtureFloat16::transpose(
        transpose::ComplexAxisLocation::complex_middle);
  }
  SECTION("complex-last") {
    TransposeTestFixtureFloat16::transpose(
        transpose::ComplexAxisLocation::complex_last);
  }
}

TEST_CASE_METHOD(TransposeTestFixtureInt1, "Transpose Test - int1",
                 "[transpose-test-int1]") {
  TransposeTestFixtureInt1::transpose(
      transpose::ComplexAxisLocation::complex_middle);
}

} // namespace ccglib::test
