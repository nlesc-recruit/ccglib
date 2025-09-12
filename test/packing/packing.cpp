#include <catch2/catch_test_macros.hpp>
#include <functional>
#include <limits.h>
#include <random>

#include <cudawrappers/cu.hpp>

#include <ccglib/packing/packing.h>

// Packing kernels do not run on AMD GPUs with ROCM < 6.2
inline void skip_if_old_rocm() {
#ifdef __HIP_PLATFORM_AMD__
  const int hip_version = HIP_VERSION_MAJOR * 10 + HIP_VERSION_MINOR;
  if (hip_version < 62) {
    SKIP("Packing kernels require ROCm 6.2+");
  }
#endif
}

namespace ccglib::test {

TEST_CASE("Packing") {
  skip_if_old_rocm();

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

  ccglib::packing::Packing packing(N, ccglib::forward, device, stream);
  packing.Run(d_in, d_out);

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
  skip_if_old_rocm();

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
  for (int i = 0; i < N / packing_factor; i++) {
    static_cast<unsigned *>(h_in)[i] = generator();
  }

  cu::DeviceMemory d_in(h_in);
  cu::DeviceMemory d_out(bytes_out);
  d_out.zero(bytes_out);

  ccglib::packing::Packing packing(N, ccglib::backward, device, stream);
  packing.Run(d_in, d_out);

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
  skip_if_old_rocm();

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

  ccglib::packing::Packing packing(N, ccglib::forward, device, stream);
  ccglib::packing::Packing unpacking(N, ccglib::backward, device, stream);

  packing.Run(d_in, d_packed);
  unpacking.Run(d_packed, d_out);

  // copy output to host
  stream.memcpyDtoHAsync(h_out, d_out, bytes_unpacked);
  stream.synchronize();

  for (size_t i = 0; i < N; i++) {
    REQUIRE(static_cast<unsigned char *>(h_in)[i] ==
            static_cast<unsigned char *>(h_out)[i]);
  }
}

TEST_CASE("Packing - complex-interleaved") {
  skip_if_old_rocm();

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

  ccglib::packing::Packing packing(N, ccglib::forward, device, stream,
                                   ccglib::complex_interleaved);
  packing.Run(d_in, d_out);

  // copy output to host
  stream.memcpyDtoHAsync(h_out, d_out, bytes_out);
  stream.synchronize();

  // verify
  for (size_t i = 0; i < N; i++) {
    // map from real0, real1, .... imag0, imag1... indexing to
    // real0, imag0, real1, imag1, ...
    size_t input_index = i * 2;
    if (input_index >= N) {
      input_index -= N - 1;
    }
    unsigned char input_value = static_cast<unsigned char *>(h_in)[input_index];
    unsigned char output_value =
        (static_cast<unsigned *>(h_out)[i / packing_factor] >>
         (i % packing_factor)) &
        1;
    REQUIRE(input_value == output_value);
  }
}

} // namespace ccglib::test
