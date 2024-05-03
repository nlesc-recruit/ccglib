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

class BeamformerTestFixture {
public:
  BeamformerTestFixture() {}

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
  void bf_gemm_and_validate() {
    cu::init();
    cu::Device device(0);
    cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, device);
    cu::Stream stream;

    // kernel settings
    const int beams_per_block = ccglib::mma::GEMM::kMPerBlock;
    const int frames_per_block = ccglib::mma::GEMM::kNPerBlock;
    const int samples_per_wmma = ccglib::mma::GEMM::kKPerWMMA;

    // data size and type, sizes match CUBE test data
    const int beams = beams_per_block;   // must be multiple of beams_per_block
    const int frames = frames_per_block; // must be multiple of frames_per_block
    const int samples =
        4 * samples_per_wmma; // must be multiple of samples_per_wmma

    using Tin = half;
    using Tout = float;

    const unsigned int nr_input_bits = sizeof(Tin) * 8;
    const unsigned int nr_output_bits = sizeof(Tout) * 8;

    const size_t bytes_a = sizeof(Tin) * COMPLEX * beams * samples;
    const size_t bytes_b = sizeof(Tin) * COMPLEX * frames * samples;
    const size_t bytes_c = sizeof(Tout) * COMPLEX * beams * frames;

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

    // run the GEMM kernel
    gemm_mma.run(d_a_trans, d_b_trans, d_c);

    // copy C to host
    stream.memcpyDtoHAsync(h_c, d_c, bytes_c);
    stream.synchronize();

    // verify output
    verify<Tin, Tout, beams, frames, samples>(static_cast<const Tin *>(h_a),
                                              static_cast<const Tin *>(h_b),
                                              static_cast<Tout *>(h_c));
  }
};

TEST_CASE_METHOD(BeamformerTestFixture, "Validate Beamformer Test",
                 "[beamform-test]") {
  BeamformerTestFixture::bf_gemm_and_validate();
}

} // namespace ccglib::test
