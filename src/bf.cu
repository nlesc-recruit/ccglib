#include <complex>
#include <cuda_fp16.h>
#include <iostream>
#include <math.h>
#include <omp.h>

#include <xtensor/xadapt.hpp>
#include <xtensor/xtensor.hpp>

#include <cudawrappers/cu.hpp>
#include <cudawrappers/nvrtc.hpp>

#include "transpose_kernel.cuh"

#include "reference/GEMM.h"
#include "transpose/Transpose.h"

#include "config.h"

#define CEILDIV(A, B) ((A) / (B) + ((A) % (B) != 0))

#ifndef COMPLEX
#define COMPLEX 2
#endif

extern const char _binary_kernels_transpose_kernel_cu_start,
    _binary_kernels_transpose_kernel_cu_end;

template <typename Tin, typename Tout, unsigned M, unsigned N, unsigned K>
void verify(const Tin *a, const Tin *b, const Tout *c) {
  const std::array<size_t, 3> a_shape = {2, M, K};
  const std::array<size_t, 3> b_shape = {2, N, K};
  const std::array<size_t, 3> c_shape = {2, M, N};

  const size_t a_size = 2 * M * K;
  const size_t b_size = 2 * N * K;
  const size_t c_size = 2 * M * N;

  auto a_view = xt::adapt(a, a_size, xt::no_ownership(), a_shape);
  auto b_view = xt::adapt(b, b_size, xt::no_ownership(), b_shape);
  auto c_view = xt::adapt(c, c_size, xt::no_ownership(), c_shape);

  xt::xtensor<Tout, 3> c_ref = xt::zeros_like(c_view);

  ccglib::reference::GEMM gemm;
  gemm.run(a, b, c_ref.data(), M, N, K);

  std::cout << "Verifying output" << std::endl;
  const int max_errs = 10;
  int errs = 0;
  for (unsigned m = 0; m < M; m++) {
    for (unsigned n = 0; n < N; n++) {
      if (errs >= max_errs) {
        break;
      }
      std::complex<Tout> ref(c_ref(0, m, n), c_ref(1, m, n));
      std::complex<Tout> tst(c_view(0, m, n), c_view(1, m, n));
      if (std::abs(ref - tst) > 1 && errs < max_errs) {
        std::cout << "Failed at m=" << m << ", n=" << n;
        std::cout << ", expected " << ref << ", found " << tst << std::endl;
        errs++;
      }
    }
  }
  if (errs == 0) {
    std::cout << "Result ok" << std::endl;
  }
}

template <typename T, typename TransT>
void transpose(cu::HostMemory &input, cu::DeviceMemory &output, size_t rows,
               size_t cols, cu::Stream &stream) {
  // Allocate device memory for non-transposed data
  cu::DeviceMemory d_input(sizeof(T));
  stream.memcpyHtoDAsync(d_input, input, sizeof(T));
  // cu::DeviceMemory d_input = stream.memAllocAsync(sizeof(T));

  dim3 threads(32, 32);
  dim3 grid(CEILDIV(rows, threads.x), CEILDIV(cols, threads.y));

  T *d_input_arg = reinterpret_cast<T *>(static_cast<CUdeviceptr>(d_input));
  TransT *d_output_arg =
      reinterpret_cast<TransT *>(static_cast<CUdeviceptr>(output));

  transpose<<<grid, threads, 0, stream>>>(*d_output_arg, *d_input_arg);
  // Ensure input is not freed before transpose finishes
  stream.synchronize();
}

int main() {
  std::cout << "Beamform main" << std::endl;

  cu::init();
  cu::Device device(0);
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, device);
  cu::Stream stream;

  // kernel settings
  const int beams_per_block = 128;
  const int beams_per_warp = 32;
  const int beams_per_wmma = 16;

  const int frames_per_block = 64;
  const int frames_per_warp = 32;
  const int frames_per_wmma = 16;

  const int samples_per_wmma = 16;
  const int warp_size = 32;
  const int nbuffer = 4;

  // data size and type, sizes match CUBE test data
  const int complex = 2;
  const int beams = 10240;  // must be multiple of beams_per_block
  const int frames = 1024;  // must be multiple of frames_per_block
  const int samples = 7808; // must be multiple of samples_per_wmma

  const int nbit = 16;
  using Tin = half;
  using Tout = float;

  // data types for matrices
  // A and B are transposed to an optimal format for the GEMM
  using A_t = Tin[COMPLEX][beams][samples];
  using B_t = Tin[COMPLEX][frames][samples];
  using A_trans_t = Tin[beams / beams_per_block][samples / samples_per_wmma]
                       [COMPLEX][beams_per_block][samples_per_wmma];
  using B_trans_t = Tin[frames / frames_per_block][samples / samples_per_wmma]
                       [COMPLEX][frames_per_block][samples_per_wmma];
  using C_t = Tout[COMPLEX][beams][frames];

  const size_t bytes_a = sizeof(A_t);
  const size_t bytes_b = sizeof(B_t);
  const size_t bytes_c = sizeof(C_t);

  // initalize host memory
  cu::HostMemory h_a(bytes_a);
  cu::HostMemory h_b(bytes_b);
  cu::HostMemory h_c(bytes_c);

  A_t *a = static_cast<A_t *>(h_a);
  B_t *b = static_cast<B_t *>(h_b);
  C_t *c = static_cast<C_t *>(h_c);

  // fill a and b with random values, initalize c to zero
  // Note: only works for Tin=half, should use e.g. KernelFloat library to
  // more easily support other types
  srand(42);
  for (int idx = 0; idx < complex * beams * samples; idx++) {
    static_cast<Tin *>(h_a)[idx] =
        __float2half(16 * ((float)rand() / RAND_MAX) - 8);
  }
  for (int idx = 0; idx < complex * frames * samples; idx++) {
    static_cast<Tin *>(h_b)[idx] =
        __float2half(16 * ((float)rand() / RAND_MAX) - 8);
  }

  // We start with a transpose kernel to get the data in the right shape for the
  // GEMM. The original data is allocated on the GPU in a subscope because we do
  // not need it anymore  after the transpose.
  // When DeviceMemory goes out of scope, the destructor will free the GPU
  // memory

  // Allocate device memory for transposed input data
  cu::DeviceMemory d_a_trans(bytes_a);
  cu::DeviceMemory d_b_trans(bytes_b);

// Transpose A
#if 0
  transpose<A_t, A_trans_t>(h_a, d_a_trans, samples, beams, stream);
#else
  ccglib::transpose::Transpose transpose_a(
      samples, beams, samples_per_wmma, beams_per_block, nbit, device, stream);
  cu::DeviceMemory d_a = stream.memAllocAsync(bytes_a);
  stream.memcpyHtoDAsync(d_a, h_a, bytes_a);
  transpose_a.run(d_a, d_a_trans);
  stream.memFreeAsync(d_a);
#endif

  // Transpose B
  transpose<B_t, B_trans_t>(h_b, d_b_trans, samples, frames, stream);

  // allocate device memory for output data and initialize to zero
  cu::DeviceMemory d_c(bytes_c);
  d_c.zero(bytes_c);

  // compile GEMM kernel
  std::string cuda_include_path = std::string(getenv("CUDA_HOME")) + "/include";
  std::string lib_include_path = std::string(INSTALL_INCLUDE_DIR);

  dim3 threads(warp_size, frames_per_block / frames_per_warp,
               beams_per_block / beams_per_warp);
  dim3 grid(CEILDIV(frames, frames_per_block), CEILDIV(beams, beams_per_block));

  std::cout << "Problem size (M, N, K): (" << beams << ", " << frames << ", "
            << samples << ")" << std::endl;
  std::cout << "Thread block size: (" << threads.x << ", " << threads.y << ", "
            << threads.z << ")" << std::endl;
  std::cout << "Threads per block: " << threads.x * threads.y * threads.z
            << std::endl;
  std::cout << "Grid size: (" << grid.x << ", " << grid.y << ", " << grid.z
            << ")" << std::endl;

  int capability =
      10 * device.getAttribute<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR>() +
      device.getAttribute<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR>();

  std::vector<std::string> options = {
      "-std=c++17",
      "-arch=sm_" + std::to_string(capability),
      "-I" + cuda_include_path,
      "-I" + lib_include_path,
      "-Dblock_size_x=" + std::to_string(threads.x),
      "-Dblock_size_y=" + std::to_string(threads.y),
      "-Dblock_size_z=" + std::to_string(threads.z),
      "-DM=" + std::to_string(beams),
      "-D_N=" + std::to_string(frames),
      "-DK=" + std::to_string(samples),
      "-DNBIT=" + std::to_string(nbit),
      "-DM_PER_BLOCK=" + std::to_string(beams_per_block),
      "-DM_PER_WARP=" + std::to_string(beams_per_warp),
      "-DM_PER_WMMA=" + std::to_string(beams_per_wmma),
      "-DN_PER_BLOCK=" + std::to_string(frames_per_block),
      "-DN_PER_WARP=" + std::to_string(frames_per_warp),
      "-DN_PER_WMMA=" + std::to_string(frames_per_wmma),
      "-DK_PER_WMMA=" + std::to_string(samples_per_wmma),
      "-DWARP_SIZE=" + std::to_string(warp_size),
      "-DNBUFFER=" + std::to_string(nbuffer)};

  // the kernel source is embedded into the object file by the linker
  // the name of these extern variables depends on the source path!
  // it is also possible to just load the kernel source file at runtime,
  // but this requires the source file to be in a known location and
  // allows the user to (accidentally) replace it by different code,
  // crashing this programme. Hence we opt for the embedded version
  extern const char _binary_kernels_gemm_kernel_cu_start,
      _binary_kernels_gemm_kernel_cu_end;
  const std::string kernel(&_binary_kernels_gemm_kernel_cu_start,
                           &_binary_kernels_gemm_kernel_cu_end);

  nvrtc::Program program(kernel, "gemm_kernel.cu");

  try {
    program.compile(options);
  } catch (nvrtc::Error &error) {
    std::cerr << program.getLog();
    throw;
  }
  std::cout << "Kernel compilation succeeded" << std::endl;

  cu::Module module(static_cast<const void *>(program.getPTX().data()));
  cu::Function function(module, "wmma_complex_gemm_opt");

  // run and time the GEMM kernel
  std::vector<const void *> parameters = {
      d_c.parameter(), d_a_trans.parameter(), d_b_trans.parameter()};

  cu::Event start, end;
  start.record(stream);
  stream.launchKernel(function, grid.x, grid.y, grid.z, threads.x, threads.y,
                      threads.z, 0, parameters);
  end.record(stream);
  end.synchronize();

  float time = end.elapsedTime(start);
  std::cout << "Kernel took " << time << " ms" << std::endl;
  float tflops = 8ULL * 1e-9 * beams * frames * samples / time;
  std::cout << "TFLOPS: " << tflops << std::endl;

  // copy C to host
  stream.memcpyDtoHAsync(c, d_c, bytes_c);
  stream.synchronize();

  // verify output
  verify<Tin, Tout, beams, frames, samples>(reinterpret_cast<const Tin *>(a),
                                            reinterpret_cast<const Tin *>(b),
                                            reinterpret_cast<Tout *>(c));
  return 0;
}