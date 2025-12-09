#include <catch2/catch_test_macros.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <limits>

#include <xtensor/xadapt.hpp>
#include <xtensor/xtensor.hpp>

#include <ccglib/common/precision.h>
#include <ccglib/fp16.h>
#include <ccglib/fp6.h>
#include <ccglib/fp8.h>
#include <ccglib/transpose/transpose.h>

namespace ccglib::test {

static inline void cuda_check(cudaError_t err) {
  if (err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
  }
}

static inline void cu_check(CUresult err) {
  if (err != CUDA_SUCCESS) {
    const char *err_str;
    cuGetErrorString(err, &err_str);
    throw std::runtime_error(err_str);
  }
}

template <typename T, ccglib::ValueType InputPrecision>
class TransposeTestFixture {
public:
  TransposeTestFixture() {
    cuda_check(cudaFree(0));
    cu_check(cuInit(0));
    cu_check(cuDeviceGet(&device_, 0));
    cu_check(cuCtxGetCurrent(&context_));
    cu_check(cuStreamCreate(&stream_, CU_STREAM_DEFAULT));
  }

  ~TransposeTestFixture() {
    cuda_check(cudaFreeHost(h_a_));
    cuda_check(cudaFreeHost(h_a_trans_));
    cu_check(cuMemFree(d_a_));
    cu_check(cuMemFree(d_a_trans_));
    cu_check(cuStreamDestroy(stream_));
  }

private:
  CUdevice device_;
  CUcontext context_;
  CUstream stream_;

  T *h_a_ = nullptr;
  T *h_a_trans_ = nullptr;
  CUdeviceptr d_a_ = 0;
  CUdeviceptr d_a_trans_ = 0;

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
    cuda_check(cudaMallocHost(&h_a_, kBytesA));
    cuda_check(cudaMallocHost(&h_a_trans_, kBytesA));
    init_input_matrix(h_a_);

    // Allocate device memory for input data
    cu_check(cuMemAlloc(&d_a_, kBytesA));

    // Transfer the input data
    cu_check(cuMemcpyHtoDAsync(d_a_, h_a_, kBytesA, stream_));

    // allocate device memory for output data and initialize to zero
    cu_check(cuMemAlloc(&d_a_trans_, kBytesA));
    cu_check(cuMemsetD8Async(d_a_trans_, 0, kBytesA, stream_));
  }

  void verify_output(ccglib::ComplexAxisLocation complex_axis_location) {
    // copy output to host
    cu_check(cuMemcpyDtoHAsync(h_a_trans_, d_a_trans_, kBytesA, stream_));
    cu_check(cuStreamSynchronize(stream_));

    // verification
    // data is transposed from shape [batch][complex][m][n / packing_factor] to
    // [batch][m/m_per_chunk][n/n_per_chunk][complex][m_per_chunk][n_per_chunk /
    // packing_factor] in complex-planar mode, where packing_factor is the
    // number of samples per item (e.g. 32 for 1-bit samples packed into 32-bit
    // ints.) In complex-interleaved mode, complex is the last axis.

    std::array<size_t, 4> shape_input;
    if (complex_axis_location == ccglib::ComplexAxisLocation::complex_planar) {
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
                ccglib::ComplexAxisLocation::complex_planar) {
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
  void transpose(ccglib::ComplexAxisLocation complex_axis_location) {
    init_memory();

    ccglib::transpose::Transpose transpose_a(
        kBatchSize, kGlobalM, kGlobalN, kMPerChunk, kNPerChunk,
        ccglib::ValuePrecision(InputPrecision).GetBitWidth(), device_, stream_,
        complex_axis_location);

    transpose_a.Run(d_a_, d_a_trans_);

    verify_output(complex_axis_location);
  }
};

using TransposeTestFixtureFloat6e2m3 =
    TransposeTestFixture<fp6_e3m2, ccglib::ValueType::float6e2m3>;
using TransposeTestFixtureFloat8e4m3 =
    TransposeTestFixture<fp8_e4m3, ccglib::ValueType::float8e4m3>;
using TransposeTestFixtureFloat16 =
    TransposeTestFixture<half, ccglib::ValueType::float16>;
using TransposeTestFixtureInt1 =
    TransposeTestFixture<unsigned int, ccglib::ValueType::int1>;

TEST_CASE_METHOD(TransposeTestFixtureFloat6e2m3, "Transpose Test - float6e2m3",
                 "[transpose-test-float6e2m3]") {
  SECTION("complex-planar") {
    transpose(ccglib::ComplexAxisLocation::complex_planar);
  }
  SECTION("complex-interleaved") {
    transpose(ccglib::ComplexAxisLocation::complex_interleaved);
  }
}

TEST_CASE_METHOD(TransposeTestFixtureFloat8e4m3, "Transpose Test - float8e4m3",
                 "[transpose-test-float8e4m3]") {
  SECTION("complex-planar") {
    transpose(ccglib::ComplexAxisLocation::complex_planar);
  }
  SECTION("complex-interleaved") {
    transpose(ccglib::ComplexAxisLocation::complex_interleaved);
  }
}

TEST_CASE_METHOD(TransposeTestFixtureFloat16, "Transpose Test - float16",
                 "[transpose-test-float16]") {
  SECTION("complex-planar") {
    transpose(ccglib::ComplexAxisLocation::complex_planar);
  }
  SECTION("complex-interleaved") {
    transpose(ccglib::ComplexAxisLocation::complex_interleaved);
  }
}

TEST_CASE_METHOD(TransposeTestFixtureInt1, "Transpose Test - int1",
                 "[transpose-test-int1]") {
  transpose(ccglib::ComplexAxisLocation::complex_planar);
}

} // namespace ccglib::test
