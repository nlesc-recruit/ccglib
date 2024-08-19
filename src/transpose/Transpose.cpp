#include <iostream>

#include <cudawrappers/nvrtc.hpp>

#include <ccglib/helper.h>
#include <ccglib/transpose/transpose.h>

extern const char _binary_kernels_transpose_kernel_cu_start,
    _binary_kernels_transpose_kernel_cu_end;

namespace ccglib::transpose {

Transpose::Transpose(size_t B, size_t M, size_t N, size_t M_chunk,
                     size_t N_chunk, size_t nr_bits, cu::Device &device,
                     cu::Stream &stream)
    : B(B), M(M), N(N), M_chunk(M_chunk), N_chunk(N_chunk), nr_bits(nr_bits),
      device_(device), stream_(stream) {
  compile_kernel();
}

void Transpose::Run(cu::HostMemory &h_input, cu::DeviceMemory &d_output) {
  cu::DeviceMemory d_input = stream_.memAllocAsync(h_input.size());
  stream_.memcpyHtoDAsync(d_input, h_input, h_input.size());
  Run(d_input, d_output);
  stream_.memFreeAsync(d_input);
}

void Transpose::Run(cu::DeviceMemory &d_input, cu::DeviceMemory &d_output) {
  dim3 threads(32, 32);
  dim3 grid(helper::ceildiv(N, threads.x), helper::ceildiv(M, threads.y), B);

  std::vector<const void *> parameters = {d_output.parameter(),
                                          d_input.parameter()};

  stream_.launchKernel(*function, grid.x, grid.y, grid.z, threads.x, threads.y,
                       threads.z, 0, parameters);
}

void Transpose::compile_kernel() {
  const std::string cuda_include_path = nvrtc::findIncludePath();

  const std::string arch = device_.getArch();

  std::vector<std::string> options = {
    "-std=c++17",
#if defined(__HIP__)
    "--offload-arch=" + arch,
#else
    "-arch=" + arch,
#endif
    "-I" + cuda_include_path,
    "-DBATCH_SIZE=" + std::to_string(B) + "UL",
    "-DM_GLOBAL=" + std::to_string(M) + "UL",
    "-DN_GLOBAL=" + std::to_string(N) + "UL",
    "-DNBIT=" + std::to_string(nr_bits),
    "-DM_CHUNK=" + std::to_string(M_chunk),
    "-DN_CHUNK=" + std::to_string(N_chunk)
  };

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
