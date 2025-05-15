#include <iostream>

#include <cudawrappers/cu.hpp>
#include <cudawrappers/nvrtc.hpp>

#include <ccglib/helper.h>
#include <ccglib/transpose/transpose.h>

extern const char _binary_kernels_transpose_kernel_cu_start,
    _binary_kernels_transpose_kernel_cu_end;

namespace ccglib::transpose {

class Transpose::Impl {
public:
  Impl(size_t B, size_t M, size_t N, size_t M_chunk, size_t N_chunk,
       size_t nr_bits, cu::Device &device, cu::Stream &stream,
       ComplexAxisLocation input_complex_axis_location);
  void Run(cu::HostMemory &h_input, cu::DeviceMemory &d_output);
  void Run(cu::DeviceMemory &d_input, cu::DeviceMemory &d_output);

private:
  size_t B;
  size_t M;
  size_t N;
  size_t M_chunk;
  size_t N_chunk;

  size_t nr_bits;
  ComplexAxisLocation input_complex_axis_location_;

  cu::Device &device_;
  cu::Stream &stream_;

  void compile_kernel();

  std::shared_ptr<cu::Module> module;
  std::shared_ptr<cu::Function> function;
};

Transpose::Impl::Impl(size_t B, size_t M, size_t N, size_t M_chunk,
                      size_t N_chunk, size_t nr_bits, cu::Device &device,
                      cu::Stream &stream,
                      ComplexAxisLocation input_complex_axis_location)
    : B(B), M(M), N(N), M_chunk(M_chunk), N_chunk(N_chunk), nr_bits(nr_bits),
      device_(device), stream_(stream),
      input_complex_axis_location_(input_complex_axis_location) {
  compile_kernel();
}

void Transpose::Impl::Run(cu::HostMemory &h_input, cu::DeviceMemory &d_output) {
  cu::DeviceMemory d_input = stream_.memAllocAsync(h_input.size());
  stream_.memcpyHtoDAsync(d_input, h_input, h_input.size());
  Run(d_input, d_output);
  stream_.memFreeAsync(d_input);
}

void Transpose::Impl::Run(cu::DeviceMemory &d_input,
                          cu::DeviceMemory &d_output) {
  const unsigned warp_size =
      device_.getAttribute(CU_DEVICE_ATTRIBUTE_WARP_SIZE);
  dim3 threads(warp_size, 1024 / warp_size);
  dim3 grid(helper::ceildiv(N, threads.x), helper::ceildiv(M, threads.y), B);

  std::vector<const void *> parameters = {d_output.parameter(),
                                          d_input.parameter()};

  stream_.launchKernel(*function, grid.x, grid.y, grid.z, threads.x, threads.y,
                       threads.z, 0, parameters);
}

void Transpose::Impl::compile_kernel() {
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
    "-DNBIT_IN=" + std::to_string(nr_bits),
    "-DM_CHUNK=" + std::to_string(M_chunk),
    "-DN_CHUNK=" + std::to_string(N_chunk)
  };

  if (input_complex_axis_location_ == ComplexAxisLocation::complex_middle) {
    options.push_back("-DINPUT_COMPLEX_MIDDLE");
  } else if (input_complex_axis_location_ ==
             ComplexAxisLocation::complex_last) {
    options.push_back("-DINPUT_COMPLEX_LAST");
  }

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

Transpose::Transpose(size_t B, size_t M, size_t N, size_t M_chunk,
                     size_t N_chunk, size_t nr_bits, cu::Device &device,
                     cu::Stream &stream,
                     ComplexAxisLocation input_complex_axis_location) {
  impl_ = std::make_unique<Impl>(B, M, N, M_chunk, N_chunk, nr_bits, device,
                                 stream, input_complex_axis_location);
}

void Transpose::Run(cu::HostMemory &h_input, cu::DeviceMemory &d_output) {
  impl_->Run(h_input, d_output);
}

void Transpose::Run(cu::DeviceMemory &d_input, cu::DeviceMemory &d_output) {
  impl_->Run(d_input, d_output);
}

Transpose::~Transpose() = default;

} // end namespace ccglib::transpose
