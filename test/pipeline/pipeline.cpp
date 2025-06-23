
#include <catch2/catch_test_macros.hpp>
#include <cudawrappers/cu.hpp>
#include <functional>
#include <limits.h>
#include <random>

#include <ccglib/fp16.h>
#include <ccglib/pipeline/pipeline.h>

#include "verify.h"

namespace ccglib::test {

TEST_CASE("Pipeline") {
  const size_t B = 1;
  const size_t M = 512;
  const size_t N = 512;
  const size_t K = 512;

  using Tin = half;
  using Tout = float;

  const size_t num_a = B * 2 * M * K;
  const size_t num_b = B * 2 * N * K;
  const size_t num_c = B * 2 * M * N;

  cu::init();
  cu::Device device(0);
  cu::Context context(CU_CTX_BLOCKING_SYNC, device);
  cu::Stream stream;

  cu::HostMemory h_a(num_a * sizeof(Tin));
  cu::HostMemory h_b(num_b * sizeof(Tin));
  cu::HostMemory h_c(num_c * sizeof(Tout));

  auto generator = std::bind(std::uniform_real_distribution<float>(-10, 10),
                             std::default_random_engine());
  for (int i = 0; i < num_a; i++) {
    static_cast<Tin *>(h_a)[i] = generator();
  }

  for (int i = 0; i < num_b; i++) {
    static_cast<Tin *>(h_b)[i] = generator();
  }

  cu::DeviceMemory d_a(h_a.size());
  cu::DeviceMemory d_b(h_b.size());
  cu::DeviceMemory d_c(h_c.size());
  stream.memcpyHtoDAsync(d_a, h_a, d_a.size());
  stream.memcpyHtoDAsync(d_b, h_b, d_b.size());
  d_c.zero(d_c.size());

  ccglib::pipeline::Pipeline pipeline(
      B, M, N, K, device, stream, ccglib::complex_planar,
      ccglib::complex_planar, ccglib::mma::row_major, ccglib::mma::col_major,
      ccglib::mma::row_major, ccglib::float16, ccglib::float32,
      ccglib::mma::opt);

  pipeline.Run(d_a, d_b, d_c);
  stream.memcpyDtoHAsync(h_c, d_c, d_c.size());
  stream.synchronize();

  verify<Tin, Tout, ccglib::float16>(
      static_cast<const Tin *>(h_a), static_cast<const Tin *>(h_b),
      static_cast<Tout *>(h_c), B, M, N, K, ccglib::mma::row_major);
}

} // namespace ccglib::test
