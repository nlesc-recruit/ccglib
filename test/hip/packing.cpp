#include <catch2/catch_test_macros.hpp>
#include <functional>
#include <limits.h>
#include <random>

#include <ccglib/packing/packing.h>
#include <hip/hip_runtime.h>

static inline void hip_check(hipError_t err) {
  if (err != hipSuccess) {
    throw std::runtime_error(hipGetErrorString(err));
  }
}

// Packing kernels do not run on AMD GPUs in HIP < 6.2
inline void skip_if_old_hip() {
#ifdef __HIP_PLATFORM_AMD__
  const int hip_version = HIP_VERSION_MAJOR * 10 + HIP_VERSION_MINOR;
  if (hip_version < 62) {
    SKIP("Packing kernels require HIP 6.2+");
  }
#endif
}

namespace ccglib::test {

TEST_CASE("HIP packing") {
  skip_if_old_hip();

  const size_t N = 2048;
  const size_t bytes_in = sizeof(unsigned char) * N;
  const size_t packing_factor =
      sizeof(unsigned) * CHAR_BIT / sizeof(unsigned char);
  const size_t bytes_out = bytes_in * sizeof(unsigned) / packing_factor;

  hip_check(hipInit(0));
  hipDevice_t device;
  hip_check(hipDeviceGet(&device, 0));
  hipStream_t stream;
  hip_check(hipStreamCreate(&stream));

  using Tin = unsigned char;
  using Tout = unsigned;

  Tin *h_in;
  Tout *h_out;
  hip_check(hipHostMalloc(&h_in, bytes_in));
  hip_check(hipHostMalloc(&h_out, bytes_out));

  auto generator = std::bind(std::uniform_int_distribution<>(0, 1),
                             std::default_random_engine());
  for (int i = 0; i < N; i++) {
    h_in[i] = generator();
  }

  Tin *d_in;
  Tout *d_out;

  hip_check(hipMalloc(&d_in, bytes_in));
  hip_check(hipMalloc(&d_out, bytes_out));
  hip_check(hipMemset(d_out, 0, bytes_out));

  hip_check(hipMemcpy(d_in, h_in, bytes_in, hipMemcpyHostToDevice));

  ccglib::packing::Packing packing(N, ccglib::forward, device, stream);
  packing.Run(reinterpret_cast<hipDeviceptr_t>(d_in),
              reinterpret_cast<hipDeviceptr_t>(d_out));

  hip_check(hipMemcpy(h_out, d_out, bytes_out, hipMemcpyDeviceToHost));

  // verify
  for (size_t i = 0; i < N; i++) {
    unsigned char input_value = h_in[i];
    unsigned char output_value =
        (h_out[i / packing_factor] >> (i % packing_factor)) & 1;
    REQUIRE(input_value == output_value);
  }

  hip_check(hipFree(d_in));
  hip_check(hipFree(d_out));

  hip_check(hipHostFree(h_in));
  hip_check(hipHostFree(h_out));
}

} // namespace ccglib::test
