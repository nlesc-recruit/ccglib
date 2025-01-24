#include "ccglib/precision.h"
#include <iostream>

#include <ccglib/ccglib.hpp>
#include <cudawrappers/cu.hpp>

int main(int argc, char *argv[]) {
  cu::init();
  cu::Device device(0);
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, device);
  cu::Stream stream;

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

  // initalize host memory
  cu::HostMemory h_a(bytes_a);
  cu::HostMemory h_b(bytes_b);
  cu::HostMemory h_c(bytes_c);

  // Allocate device memory for input data
  cu::DeviceMemory d_a(bytes_a);
  cu::DeviceMemory d_b(bytes_b);

  // Transfer the input data
  stream.memcpyHtoDAsync(d_a, h_a, bytes_a);
  stream.memcpyHtoDAsync(d_b, h_b, bytes_b);

  // allocate device memory for output data and initialize to zero
  cu::DeviceMemory d_c(bytes_c);
  d_c.zero(bytes_c);

  ccglib::mma::GEMM gemm_mma(batch_size, global_m, global_n, global_k,
                             nr_input_bits, device, stream,
                             ccglib::ValueType::float16, ccglib::mma::opt);

  // run the GEMM kernel
  cu::Event start, end;
  stream.record(start);
  gemm_mma.Run(d_a, d_b, d_c);
  stream.record(end);

  // copy C to host
  stream.memcpyDtoHAsync(h_c, d_c, bytes_c);
  stream.synchronize();

  // print runtime
  const float runtime = end.elapsedTime(start);
  const double tflops = 8ULL * 1e-9 * global_m * global_n * global_k / runtime;
  std::cout << "runtime: " << runtime << " ms" << std::endl;
  std::cout << "TFLOPS: " << tflops << std::endl;
}
