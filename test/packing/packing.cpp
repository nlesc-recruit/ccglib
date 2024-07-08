#include <catch2/catch_test_macros.hpp>
#include <functional>
#include <limits.h>
#include <random>

#include <ccglib/packing/packing.h>

namespace ccglib::test {

TEST_CASE("Packing") {
  const size_t N = 2048;
  const size_t bytes_in = sizeof(unsigned char) * N;
  const size_t packing_factor =
      sizeof(unsigned) * CHAR_BIT / sizeof(unsigned char);
  const size_t bytes_out = bytes_in * sizeof(unsigned) / packing_factor;

  cu::init();
  cu::Device device(0);
  cu::Context context(CU_CTX_BLOCKING_SYNC, device);
  cu::Stream stream;
  cu::HostMemory h_in(bytes_in);
  cu::HostMemory h_out(bytes_out);

  auto generator = std::bind(std::uniform_int_distribution<>(0, 1),
                             std::default_random_engine());
  for (int i = 0; i < N; i++) {
    static_cast<unsigned char *>(h_in)[i] = generator();
  }

  cu::DeviceMemory d_in(h_in);
  cu::DeviceMemory d_out(bytes_out);
  d_out.zero(bytes_out);

  ccglib::packing::Packing packing(N, device, stream);
  packing.Run(d_in, d_out, ccglib::packing::pack);

  // copy output to host
  stream.memcpyDtoHAsync(h_out, d_out, bytes_out);
  stream.synchronize();

  // verify
  for (size_t i = 0; i < N; i++) {
    unsigned char input_value = static_cast<unsigned char *>(h_in)[i];
    unsigned char output_value =
        (static_cast<unsigned *>(h_out)[i / packing_factor] >>
         (i % packing_factor)) &
        1;
    REQUIRE(input_value == output_value);
  }
}

TEST_CASE("Unpacking") {
  const size_t N = 2048;
  const size_t bytes_out = sizeof(unsigned char) * N;
  const size_t packing_factor =
      sizeof(unsigned) * CHAR_BIT / sizeof(unsigned char);
  const size_t bytes_in = bytes_out * sizeof(unsigned) / packing_factor;

  cu::init();
  cu::Device device(0);
  cu::Context context(CU_CTX_BLOCKING_SYNC, device);
  cu::Stream stream;
  cu::HostMemory h_in(bytes_in);
  cu::HostMemory h_out(bytes_out);

  auto generator = std::bind(std::uniform_int_distribution<>(0, UINT32_MAX),
                             std::default_random_engine());
  for (int i = 0; i < N; i++) {
    static_cast<unsigned *>(h_in)[i] = generator();
  }

  cu::DeviceMemory d_in(h_in);
  cu::DeviceMemory d_out(bytes_out);
  d_out.zero(bytes_out);

  ccglib::packing::Packing packing(N, device, stream);
  packing.Run(d_in, d_out, ccglib::packing::unpack);

  // copy output to host
  stream.memcpyDtoHAsync(h_out, d_out, bytes_out);
  stream.synchronize();

  // verify
  for (size_t i = 0; i < N; i++) {
    unsigned char input_value =
        (static_cast<unsigned *>(h_in)[i / packing_factor] >>
         (i % packing_factor)) &
        1;
    unsigned char output_value = static_cast<unsigned char *>(h_out)[i];
    REQUIRE(input_value == output_value);
  }
}

TEST_CASE("Pack - unpack") {
  const size_t N = 2048;
  const size_t bytes_unpacked = sizeof(unsigned char) * N;
  const size_t packing_factor =
      sizeof(unsigned) * CHAR_BIT / sizeof(unsigned char);
  const size_t bytes_packed =
      bytes_unpacked * sizeof(unsigned) / packing_factor;

  cu::init();
  cu::Device device(0);
  cu::Context context(CU_CTX_BLOCKING_SYNC, device);
  cu::Stream stream;
  cu::HostMemory h_in(bytes_unpacked);
  cu::HostMemory h_out(bytes_unpacked);

  auto generator = std::bind(std::uniform_int_distribution<>(0, 1),
                             std::default_random_engine());
  for (size_t i = 0; i < N; i++) {
    static_cast<unsigned char *>(h_in)[i] = generator();
  }

  cu::DeviceMemory d_in(h_in);
  cu::DeviceMemory d_packed(bytes_packed);
  cu::DeviceMemory d_out(bytes_unpacked);
  d_packed.zero(bytes_packed);
  d_out.zero(bytes_packed);

  ccglib::packing::Packing packing(N, device, stream);

  packing.Run(d_in, d_packed, ccglib::packing::pack);
  packing.Run(d_packed, d_out, ccglib::packing::unpack);

  // copy output to host
  stream.memcpyDtoHAsync(h_out, d_out, bytes_unpacked);
  stream.synchronize();

  for (size_t i = 0; i < N; i++) {
    REQUIRE(static_cast<unsigned char *>(h_in)[i] ==
            static_cast<unsigned char *>(h_out)[i]);
  }
}

} // namespace ccglib::test
