#include <catch2/catch_test_macros.hpp>

#include <ccglib/gemm/mma.h>
#include <hip/hip_runtime.h>

#include "fpequals.h"

static inline void hip_check(hipError_t err) {
  if (err != hipSuccess) {
    throw std::runtime_error(hipGetErrorString(err));
  }
}

namespace ccglib::test {

TEST_CASE("HIP mma") {
  hip_check(hipInit(0));
  hipDevice_t device;
  hip_check(hipDeviceGet(&device, 0));
  hipStream_t stream;
  hip_check(hipStreamCreate(&stream));

  const size_t global_m = 512;
  const size_t global_n = 512;
  const size_t global_k = 512;
  const size_t batch_size = 3;
  const size_t COMPLEX = 2;

  using Tin = half;
  using Tout = float;

  const size_t nr_input_bits = sizeof(Tin) * 8;

  ccglib::mma::GEMM gemm(batch_size, global_m, global_n, global_k,
                         nr_input_bits, device, stream,
                         ccglib::ValueType::float16, ccglib::mma::basic);

  const size_t bytes_a =
      sizeof(Tin) * batch_size * COMPLEX * global_m * global_k;
  const size_t bytes_b =
      sizeof(Tin) * batch_size * COMPLEX * global_n * global_k;
  const size_t bytes_c =
      sizeof(Tout) * batch_size * COMPLEX * global_m * global_n;

  Tin *h_a;
  Tin *h_b;
  Tout *h_c;

  hip_check(hipHostMalloc(&h_a, bytes_a));
  hip_check(hipHostMalloc(&h_b, bytes_b));
  hip_check(hipHostMalloc(&h_c, bytes_c));

  Tin *d_a;
  Tin *d_b;
  Tout *d_c;

  hip_check(hipMalloc(&d_a, bytes_a));
  hip_check(hipMalloc(&d_b, bytes_b));
  hip_check(hipMalloc(&d_c, bytes_c));

  for (size_t i = 0; i < batch_size * COMPLEX * global_m * global_k; i++) {
    h_a[i] = static_cast<Tin>(1);
  }

  for (size_t i = 0; i < batch_size * COMPLEX * global_n * global_k; i++) {
    h_b[i] = static_cast<Tin>(1);
  }

  hip_check(hipMemcpy(d_a, h_a, bytes_a, hipMemcpyHostToDevice));
  hip_check(hipMemcpy(d_b, h_b, bytes_b, hipMemcpyHostToDevice));

  gemm.Run(reinterpret_cast<hipDeviceptr_t>(d_a),
           reinterpret_cast<hipDeviceptr_t>(d_b),
           reinterpret_cast<hipDeviceptr_t>(d_c));

  hip_check(hipMemcpy(h_c, d_c, bytes_c, hipMemcpyDeviceToHost));

  // with all inputs one, the C matrix has zero for the real part and 2*K for
  // the imaginary part
  for (size_t i = 0; i < batch_size * COMPLEX * global_m * global_n; i++) {
    const size_t index = i % (global_m * global_n);
    const Tout expected_value =
        index < global_m * global_n ? 0.0f : 2.0f * global_k;
    ccglib::test::fpEquals(h_c[index], expected_value);
  }

  hip_check(hipFree(d_a));
  hip_check(hipFree(d_b));
  hip_check(hipFree(d_c));

  hip_check(hipHostFree(h_a));
  hip_check(hipHostFree(h_b));
  hip_check(hipHostFree(h_c));
}

} // namespace ccglib::test
