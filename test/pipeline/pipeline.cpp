
#include <catch2/catch_test_macros.hpp>
#include <cudawrappers/cu.hpp>
#include <functional>
#include <limits.h>
#include <random>

#include <ccglib/common/helper.h>
#include <ccglib/packing/packing.h>
#include <ccglib/pipeline/pipeline.h>

#include "verify.h"

namespace ccglib::test {

TEST_CASE("Pipeline float16 - float32") {
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

TEST_CASE("Pipeline int1 - int32") {
#ifdef __HIP_PLATFORM_AMD__
  SKIP("int1 is not available on AMD GPUs");
#endif

  const size_t B = 2;
  const size_t M = 512;
  const size_t N = 512;
  const size_t K = 512;

  using Tin = unsigned char;
  using Tpacked = unsigned;
  using Tout = int;
  const ccglib::ValueType input_precision = ccglib::int1;
  const ccglib::ValueType output_precision = ccglib::int32;
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

  const size_t bytes_a_packed =
      sizeof(Tpacked) *
      ccglib::helper::ceildiv(num_a / CHAR_BIT, sizeof(Tpacked));
  const size_t bytes_a = bytes_a_packed * CHAR_BIT;
  const size_t bytes_b_packed =
      sizeof(Tpacked) *
      ccglib::helper::ceildiv(num_b / CHAR_BIT, sizeof(Tpacked));
  const size_t bytes_b = bytes_b_packed * CHAR_BIT;

  cu::init();
  cu::Device device(0);
  cu::Context context(CU_CTX_BLOCKING_SYNC, device);
  cu::Stream stream;

  cu::HostMemory h_a_packed(bytes_a_packed);
  cu::HostMemory h_b_packed(bytes_b_packed);
  cu::HostMemory h_c(num_c * sizeof(Tout));

  // the reference GEMM used in output verification does not support packing.
  // As a workaround, we generate random packed data and use the unpacking
  // kernel to create the data given as input to the pipeline.
  auto generator =
      std::bind(std::uniform_int_distribution<Tpacked>(0, UINT_MAX),
                std::default_random_engine());
  for (int i = 0; i < bytes_a_packed / sizeof(Tpacked); i++) {
    static_cast<Tpacked *>(h_a_packed)[i] = generator();
  }

  for (int i = 0; i < bytes_b_packed / sizeof(Tpacked); i++) {
    static_cast<Tpacked *>(h_b_packed)[i] = generator();
  }

  cu::DeviceMemory d_a_packed(h_a_packed.size());
  cu::DeviceMemory d_b_packed(h_b_packed.size());
  cu::DeviceMemory d_a(bytes_a);
  cu::DeviceMemory d_b(bytes_b);

  stream.memcpyHtoDAsync(d_a_packed, h_a_packed, d_a_packed.size());
  stream.memcpyHtoDAsync(d_b_packed, h_b_packed, d_b_packed.size());

  ccglib::packing::Packing unpack_a(num_a, ccglib::backward, device, stream,
                                    input_complex_axis_location);
  ccglib::packing::Packing unpack_b(num_a, ccglib::backward, device, stream,
                                    input_complex_axis_location);
  unpack_a.Run(d_a_packed, d_a);
  unpack_b.Run(d_b_packed, d_b);

  cu::DeviceMemory d_c(h_c.size());
  d_c.zero(d_c.size());

  ccglib::pipeline::Pipeline pipeline(
      B, M, N, K, device, stream, input_complex_axis_location,
      output_complex_axis_location, a_mem_order, b_mem_order, c_mem_order,
      input_precision, output_precision, variant);

  pipeline.Run(d_a, d_b, d_c);
  stream.memcpyDtoHAsync(h_c, d_c, d_c.size());
  stream.synchronize();

  verify<Tpacked, Tout, input_precision>(
      static_cast<const Tpacked *>(h_a_packed),
      static_cast<const Tpacked *>(h_b_packed), static_cast<Tout *>(h_c), B, M,
      N, K, c_mem_order);
}

} // namespace ccglib::test
