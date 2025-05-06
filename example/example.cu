#include <iostream>

#include <ccglib/ccglib.hpp>
#include <cuda.h>
#include <cuda_runtime.h>

inline void cu_check(CUresult err) {
  if (err != CUDA_SUCCESS) {
    const char *err_str;
    cuGetErrorString(err, &err_str);
    std::cerr << "CUDA driver error: " << err_str << std::endl;
    exit(1);
  }
}

inline void cuda_check(cudaError_t err) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    exit(1);
  }
}

int main() {
  // Ensure the runtime API creates a context
  cuda_check(cudaFree(0));

  // Initialize driver API for access to Device object
  CUcontext context;
  cu_check(cuCtxGetCurrent(&context));
  CUdevice device;
  cu_check(cuDeviceGet(&device, 0));
  CUstream stream;
  cu_check(cuStreamCreate(&stream, CU_STREAM_DEFAULT));

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

  cuda_check(cudaMallocHost(&h_a, bytes_a));
  cuda_check(cudaMallocHost(&h_b, bytes_b));
  cuda_check(cudaMallocHost(&h_c, bytes_c));

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
  cuda_check(cudaMalloc(&d_a, bytes_a));
  cuda_check(cudaMalloc(&d_b, bytes_b));

  // Transfer the input data
  cuda_check(cudaMemcpy(d_a, h_a, bytes_a, cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(d_b, h_b, bytes_b, cudaMemcpyHostToDevice));

  // Allocate device memory for output data and initialize to zero
  Tout *d_c;
  cuda_check(cudaMalloc(&d_c, bytes_c));
  cuda_check(cudaMemset(d_c, 0, bytes_c));

  ccglib::mma::GEMM gemm_mma(batch_size, global_m, global_n, global_k,
                             nr_input_bits, device, stream,
                             ccglib::ValueType::float16, ccglib::mma::opt);

  // Run the GEMM kernel
  cudaEvent_t start, end;
  cuda_check(cudaEventCreate(&start));
  cuda_check(cudaEventCreate(&end));
  cuda_check(cudaEventRecord(start));
  gemm_mma.Run(reinterpret_cast<CUdeviceptr>(d_a),
               reinterpret_cast<CUdeviceptr>(d_b),
               reinterpret_cast<CUdeviceptr>(d_c));
  cuda_check(cudaEventRecord(end));
  cuda_check(cudaEventSynchronize(end));

  // Copy C to host
  cuda_check(cudaMemcpy(h_c, d_c, bytes_c, cudaMemcpyDeviceToHost));

  // Print runtime
  float runtime;
  cuda_check(cudaEventElapsedTime(&runtime, start, end));
  const double tflops = 8ULL * 1e-9 * global_m * global_n * global_k / runtime;
  std::cout << "runtime: " << runtime << " ms" << std::endl;
  std::cout << "TFLOPS: " << tflops << std::endl;

  // Cleanup
  cuda_check(cudaFree(d_a));
  cuda_check(cudaFree(d_b));
  cuda_check(cudaFree(d_c));

  cuda_check(cudaFreeHost(h_a));
  cuda_check(cudaFreeHost(h_b));
  cuda_check(cudaFreeHost(h_c));
}
