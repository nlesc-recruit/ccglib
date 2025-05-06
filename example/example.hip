#include <iostream>

#include <hip/hip_runtime.h>
#include <ccglib/ccglib.hpp>

inline void hip_check(hipError_t err) {
  if (err != hipSuccess) {
    std::cerr << "HIP error: " << hipGetErrorString(err) << std::endl;
    exit(1);
  }
}

int main() {
  hip_check(hipInit(0));
  hipDevice_t device;
  hip_check(hipDeviceGet(&device, 0));
  hipStream_t stream;
  hip_check(hipStreamCreate(&stream));

  // data size and type
  const int global_m = 8192;
  const int global_n = 8192;
  const int global_k = 8192;
  const int batch_size = 1;
  const int COMPLEX = 2;

  using Tin = half;
  using Tout = float;

  const size_t nr_input_bits = sizeof(Tin) * 8;

  const size_t bytes_a =
      sizeof(Tin) * batch_size * COMPLEX * global_m * global_k;
  const size_t bytes_b =
      sizeof(Tin) * batch_size * COMPLEX * global_n * global_k;
  const size_t bytes_c =
      sizeof(Tout) * batch_size * COMPLEX * global_m * global_n;

  // Allocate host memory
  Tin *h_a;
  Tin *h_b;
  Tout *h_c;

  hip_check(hipHostMalloc(&h_a, bytes_a));
  hip_check(hipHostMalloc(&h_b, bytes_b));
  hip_check(hipHostMalloc(&h_c, bytes_c));

  // Initialize host data
  for (size_t i = 0; i < batch_size * COMPLEX * global_m * global_k; i++) {
    h_a[i] = static_cast<Tin>(1);
  }

  for (size_t i = 0; i < batch_size * COMPLEX * global_n * global_k; i++) {
    h_b[i] = static_cast<Tin>(1);
  }

  // Allocate device memory for input data
  Tin *d_a;
  Tin *d_b;
  hip_check(hipMalloc(&d_a, bytes_a));
  hip_check(hipMalloc(&d_b, bytes_b));

  // Transfer the input data
  hip_check(hipMemcpy(d_a, h_a, bytes_a, hipMemcpyHostToDevice));
  hip_check(hipMemcpy(d_b, h_b, bytes_b, hipMemcpyHostToDevice));

  // Allocate device memory for output data and initialize to zero
  Tout *d_c;
  hip_check(hipMalloc(&d_c, bytes_c));
  hip_check(hipMemset(d_c, 0, bytes_c));

  ccglib::mma::GEMM gemm_mma(batch_size, global_m, global_n, global_k,
                             nr_input_bits, device, stream,
                             ccglib::ValueType::float16, ccglib::mma::opt);

  // Run the GEMM kernel
  hipEvent_t start, end;
  hip_check(hipEventCreate(&start));
  hip_check(hipEventCreate(&end));
  hip_check(hipEventRecord(start));
  gemm_mma.Run(reinterpret_cast<hipDeviceptr_t>(d_a),
               reinterpret_cast<hipDeviceptr_t>(d_b),
               reinterpret_cast<hipDeviceptr_t>(d_c));
  hip_check(hipEventRecord(end));
  hip_check(hipEventSynchronize(end));

  // Copy C to host
  hip_check(hipMemcpy(h_c, d_c, bytes_c, hipMemcpyDeviceToHost));

  // Print runtime
  float runtime;
  hip_check(hipEventElapsedTime(&runtime, start, end));
  const double tflops = 8ULL * 1e-9 * global_m * global_n * global_k / runtime;
  std::cout << "runtime: " << runtime << " ms" << std::endl;
  std::cout << "TFLOPS: " << tflops << std::endl;

  // Cleanup
  hip_check(hipFree(d_a));
  hip_check(hipFree(d_b));
  hip_check(hipFree(d_c));

  hip_check(hipFreeHost(h_a));
  hip_check(hipFreeHost(h_b));
  hip_check(hipFreeHost(h_c));
}
