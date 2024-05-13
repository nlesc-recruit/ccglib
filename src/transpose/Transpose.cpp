#include <iostream>

#include <cudawrappers/nvrtc.hpp>

#include "Transpose.h"
#include "helper/helper.h"

extern const char _binary_kernels_transpose_kernel_cu_start,
    _binary_kernels_transpose_kernel_cu_end;

namespace ccglib::transpose {

Transpose::Transpose(size_t B, size_t M, size_t N, size_t M_chunk,
                     size_t N_chunk, size_t nr_bits, cu::Device &device,
                     cu::Stream &stream)
    : B(B), M(M), N(N), M_chunk(M_chunk), N_chunk(N_chunk), nr_bits(nr_bits),
      device(device), stream(stream) {
  compile_kernel();
}

void Transpose::run(cu::HostMemory &h_input, cu::DeviceMemory &d_output) {
  cu::DeviceMemory d_input = stream.memAllocAsync(h_input.size());
  stream.memcpyHtoDAsync(d_input, h_input, h_input.size());
  run(d_input, d_output);
  stream.memFreeAsync(d_input);
}

void Transpose::run(cu::DeviceMemory &d_input, cu::DeviceMemory &d_output) {
  dim3 threads(32, 32);
  dim3 grid(helper::ceildiv(N, threads.x), helper::ceildiv(M, threads.y), B);

  std::vector<const void *> parameters = {d_output.parameter(),
                                          d_input.parameter()};

  stream.launchKernel(*function, grid.x, grid.y, grid.z, threads.x, threads.y,
                      threads.z, 0, parameters);
}

void Transpose::compile_kernel() {
  const std::string cuda_include_path =
      std::string(getenv("CUDA_HOME")) + "/include";

  const int capability = helper::get_capability(device);

  std::vector<std::string> options = {"-std=c++17",
                                      "-arch=sm_" + std::to_string(capability),
                                      "-I" + cuda_include_path,
                                      "-DB=" + std::to_string(B),
                                      "-DM=" + std::to_string(M),
                                      "-DN=" + std::to_string(N),
                                      "-DNBIT=" + std::to_string(nr_bits),
                                      "-DM_CHUNK=" + std::to_string(M_chunk),
                                      "-DN_CHUNK=" + std::to_string(N_chunk)};

  const std::string kernel(&_binary_kernels_transpose_kernel_cu_start,
                           &_binary_kernels_transpose_kernel_cu_end);

  nvrtc::Program program(kernel, "transpose_kernel.cu");

  try {
    program.compile(options);
  } catch (nvrtc::Error &error) {
    std::cerr << program.getLog();
    throw;
  }

  module = std::make_unique<cu::Module>(
      static_cast<const void *>(program.getPTX().data()));
  function = std::make_unique<cu::Function>(*module, "transpose");
}
} // end namespace ccglib::transpose
