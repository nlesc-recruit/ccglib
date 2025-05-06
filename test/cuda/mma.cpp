#include <catch2/catch_test_macros.hpp>

#include <ccglib/gemm/mma.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "fpequals.h"

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

TEST_CASE("CUDA mma") {
  cuda_check(cudaFree(0));
  CUcontext context;
  cu_check(cuCtxGetCurrent(&context));
  CUdevice device;
  cu_check(cuDeviceGet(&device, 0));
  CUstream stream;
  cu_check(cuStreamCreate(&stream, CU_STREAM_DEFAULT));

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

  cuda_check(cudaMallocHost(&h_a, bytes_a));
  cuda_check(cudaMallocHost(&h_b, bytes_b));
  cuda_check(cudaMallocHost(&h_c, bytes_c));

  Tin *d_a;
  Tin *d_b;
  Tout *d_c;

  cuda_check(cudaMalloc(&d_a, bytes_a));
  cuda_check(cudaMalloc(&d_b, bytes_b));
  cuda_check(cudaMalloc(&d_c, bytes_c));

  for (size_t i = 0; i < batch_size * COMPLEX * global_m * global_k; i++) {
    h_a[i] = static_cast<Tin>(1);
  }

  for (size_t i = 0; i < batch_size * COMPLEX * global_n * global_k; i++) {
    h_b[i] = static_cast<Tin>(1);
  }

  cuda_check(cudaMemcpy(d_a, h_a, bytes_a, cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(d_b, h_b, bytes_b, cudaMemcpyHostToDevice));

  gemm.Run(reinterpret_cast<CUdeviceptr>(d_a),
           reinterpret_cast<CUdeviceptr>(d_b),
           reinterpret_cast<CUdeviceptr>(d_c));

  cuda_check(cudaMemcpy(h_c, d_c, bytes_c, cudaMemcpyDeviceToHost));

  // with all inputs one, the C matrix has zero for the real part and 2*K for
  // the imaginary part
  for (size_t i = 0; i < batch_size * COMPLEX * global_m * global_n; i++) {
    const size_t index = i % (global_m * global_n);
    const Tout expected_value =
        index < global_m * global_n ? 0.0f : 2.0f * global_k;
    ccglib::test::fpEquals(h_c[index], expected_value);
  }

  cuda_check(cudaFree(d_a));
  cuda_check(cudaFree(d_b));
  cuda_check(cudaFree(d_c));

  cuda_check(cudaFreeHost(h_a));
  cuda_check(cudaFreeHost(h_b));
  cuda_check(cudaFreeHost(h_c));
}

} // namespace ccglib::test
