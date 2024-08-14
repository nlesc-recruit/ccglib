#include <iostream>

#include <cudawrappers/nvrtc.hpp>

#include <ccglib/helper.h>
#include <ccglib/packing/packing.h>

extern const char _binary_kernels_packing_kernel_cu_start,
    _binary_kernels_packing_kernel_cu_end;

namespace ccglib::packing {

Packing::Packing(size_t N, cu::Device &device, cu::Stream &stream)
    : N_(N), device_(device), stream_(stream) {
  compile_kernel();
}

void Packing::Run(cu::HostMemory &h_input, cu::DeviceMemory &d_output,
                  Direction direction,
                  ComplexAxisLocation complex_axis_location) {
  cu::DeviceMemory d_input = stream_.memAllocAsync(h_input.size());
  stream_.memcpyHtoDAsync(d_input, h_input, h_input.size());
  Run(d_input, d_output, direction);
  stream_.memFreeAsync(d_input);
}

void Packing::Run(cu::DeviceMemory &d_input, cu::DeviceMemory &d_output,
                  Direction direction,
                  ComplexAxisLocation complex_axis_location) {
  dim3 threads(256);
  dim3 grid(helper::ceildiv(N_, threads.x));

  bool complex_axis_is_last =
      complex_axis_location == ComplexAxisLocation::complex_last;

  std::vector<const void *> parameters = {d_output.parameter(),
                                          d_input.parameter()};

  std::shared_ptr<cu::Function> function;
  switch (direction) {
  case Direction::pack:
    function = function_pack_;
    // complex-last is only supported in the packing kernel
    parameters.push_back(static_cast<const void *>(&complex_axis_is_last));
    break;
  case Direction::unpack:
    function = function_unpack_;
    // complex-last is not supported in the unpacking kernel
    if (complex_axis_is_last) {
      throw std::runtime_error(
          "complex-last input is not supported in unpacking kernel");
    }
    break;
  default:
    throw std::runtime_error("Unknown packing direction");
  }

  stream_.launchKernel(*function, grid.x, grid.y, grid.z, threads.x, threads.y,
                       threads.z, 0, parameters);
}

void Packing::compile_kernel() {
  const std::string cuda_include_path = nvrtc::findIncludePath();

  const int capability = helper::get_capability(device_);

  std::vector<std::string> options = {
      "-std=c++17", "-arch=sm_" + std::to_string(capability),
      "-I" + cuda_include_path, "-DN=" + std::to_string(N_) + "UL"};

  const std::string kernel(&_binary_kernels_packing_kernel_cu_start,
                           &_binary_kernels_packing_kernel_cu_end);

  nvrtc::Program program(kernel, "packing_kernel.cu");

  try {
    program.compile(options);
  } catch (nvrtc::Error &error) {
    std::cerr << program.getLog();
    throw;
  }

  module_ = std::make_unique<cu::Module>(
      static_cast<const void *>(program.getPTX().data()));
  function_pack_ = std::make_unique<cu::Function>(*module_, "pack_bits");
  function_unpack_ = std::make_unique<cu::Function>(*module_, "unpack_bits");
}
} // end namespace ccglib::packing
