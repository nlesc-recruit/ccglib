#include <catch2/catch_test_macros.hpp>
#include <functional>
#include <limits.h>
#include <random>

#include <ccglib/packing/packing.h>
#include <cuda.h>
#include <cuda_runtime.h>

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

namespace ccglib::test {

TEST_CASE("CUDA packing") {
  const size_t N = 2048;
  const size_t bytes_in = sizeof(unsigned char) * N;
  const size_t packing_factor =
      sizeof(unsigned) * CHAR_BIT / sizeof(unsigned char);
  const size_t bytes_out = bytes_in * sizeof(unsigned) / packing_factor;

  cuda_check(cudaFree(0));
  CUcontext context;
  cu_check(cuCtxGetCurrent(&context));
  CUdevice device;
  cu_check(cuDeviceGet(&device, 0));
  CUstream stream;
  cu_check(cuStreamCreate(&stream, CU_STREAM_DEFAULT));

  using Tin = unsigned char;
  using Tout = unsigned;

  Tin *h_in;
  Tout *h_out;
  cuda_check(cudaMallocHost(&h_in, bytes_in));
  cuda_check(cudaMallocHost(&h_out, bytes_out));

  auto generator = std::bind(std::uniform_int_distribution<>(0, 1),
                             std::default_random_engine());
  for (int i = 0; i < N; i++) {
    h_in[i] = generator();
  }

  Tin *d_in;
  Tout *d_out;

  cuda_check(cudaMalloc(&d_in, bytes_in));
  cuda_check(cudaMalloc(&d_out, bytes_out));
  cuda_check(cudaMemset(d_out, 0, bytes_out));

  cuda_check(cudaMemcpy(d_in, h_in, bytes_in, cudaMemcpyHostToDevice));

  ccglib::packing::Packing packing(N, ccglib::forward, device, stream);
  packing.Run(reinterpret_cast<CUdeviceptr>(d_in),
              reinterpret_cast<CUdeviceptr>(d_out));

  cuda_check(cudaMemcpy(h_out, d_out, bytes_out, cudaMemcpyDeviceToHost));

  // verify
  for (size_t i = 0; i < N; i++) {
    unsigned char input_value = h_in[i];
    unsigned char output_value =
        (h_out[i / packing_factor] >> (i % packing_factor)) & 1;
    REQUIRE(input_value == output_value);
  }

  cuda_check(cudaFree(d_in));
  cuda_check(cudaFree(d_out));

  cuda_check(cudaFreeHost(h_in));
  cuda_check(cudaFreeHost(h_out));
}

} // namespace ccglib::test
