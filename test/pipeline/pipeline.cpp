
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
  const size_t B = 2;
  const size_t M = 300;
  const size_t N = 200;
  const size_t K = 100;

  using Tin = half;
  using Tout = float;
  const ccglib::ValueType input_precision = ccglib::float16;
  const ccglib::ValueType output_precision = ccglib::float32;
  const ccglib::ComplexAxisLocation input_complex_axis_location =
      ccglib::complex_planar;
  const ccglib::ComplexAxisLocation output_complex_axis_location =
      ccglib::complex_planar;
  const ccglib::mma::MemOrder a_mem_order = ccglib::mma::row_major;
  const ccglib::mma::MemOrder b_mem_order = ccglib::mma::col_major;
  const ccglib::mma::MemOrder c_mem_order = ccglib::mma::row_major;
  const ccglib::mma::Variant variant = ccglib::mma::opt;

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
      B, M, N, K, device, stream, input_complex_axis_location,
      output_complex_axis_location, a_mem_order, b_mem_order, c_mem_order,
      input_precision, output_precision, variant);

  pipeline.Run(d_a, d_b, d_c);
  stream.memcpyDtoHAsync(h_c, d_c, d_c.size());
  stream.synchronize();

  verify<Tin, Tout, input_precision>(
      static_cast<const Tin *>(h_a), static_cast<const Tin *>(h_b),
      static_cast<Tout *>(h_c), B, M, N, K, c_mem_order);
}

} // namespace ccglib::test
