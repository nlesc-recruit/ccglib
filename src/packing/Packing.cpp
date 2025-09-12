#include <iostream>

#include <cudawrappers/cu.hpp>
#include <cudawrappers/nvrtc.hpp>

#include <ccglib/common/helper.h>
#include <ccglib/packing/packing.h>

extern const char _binary_kernels_packing_kernel_cu_start,
    _binary_kernels_packing_kernel_cu_end;

namespace ccglib::packing {

class Packing::Impl {
public:
  Impl(size_t N, Direction direction, cu::Device &device, cu::Stream &stream,
       ComplexAxisLocation input_complex_axis_location);

  void Run(cu::HostMemory &h_input, cu::DeviceMemory &d_output);

  void Run(cu::DeviceMemory &h_input, cu::DeviceMemory &d_output);

private:
  void compile_kernel();

  size_t N_;
  Direction direction_;
  cu::Device &device_;
  cu::Stream &stream_;
  ComplexAxisLocation input_complex_axis_location_;

  std::shared_ptr<cu::Module> module_;
  std::shared_ptr<cu::Function> function_;
};

Packing::Impl::Impl(size_t N, Direction direction, cu::Device &device,
                    cu::Stream &stream,
                    ComplexAxisLocation input_complex_axis_location)
    : N_(N), direction_(direction), device_(device), stream_(stream),
      input_complex_axis_location_(input_complex_axis_location) {
  // Packing kernel uses functions not supported in HIP < 6.2
#if defined(__HIP_PLATFORM_AMD__)
  const int hip_version = HIP_VERSION_MAJOR * 10 + HIP_VERSION_MINOR;
  if (hip_version < 62) {
    throw std::runtime_error("packing kernel requires HIP 6.2+");
  }
#endif

  if (direction == Direction::backward &&
      input_complex_axis_location == ComplexAxisLocation::complex_interleaved) {
    throw std::runtime_error(
        "complex-interleaved input is not supported in unpacking kernel");
  }

  compile_kernel();
}

void Packing::Impl::Run(cu::HostMemory &h_input, cu::DeviceMemory &d_output) {
  cu::DeviceMemory d_input = stream_.memAllocAsync(h_input.size());
  stream_.memcpyHtoDAsync(d_input, h_input, h_input.size());
  Run(d_input, d_output);
  stream_.memFreeAsync(d_input);
}

void Packing::Impl::Run(cu::DeviceMemory &d_input, cu::DeviceMemory &d_output) {
  dim3 threads(256);
  dim3 grid(helper::ceildiv(N_, threads.x));

  std::vector<const void *> parameters = {d_output.parameter(),
                                          d_input.parameter()};

  stream_.launchKernel(*function_, grid.x, grid.y, grid.z, threads.x, threads.y,
                       threads.z, 0, parameters);
}

void Packing::Impl::compile_kernel() {
  const std::string cuda_include_path = nvrtc::findIncludePath();

  const std::string arch = device_.getArch();
  const unsigned warp_size =
      device_.getAttribute(CU_DEVICE_ATTRIBUTE_WARP_SIZE);

  std::vector<std::string> options = {
    "-std=c++17",
#if defined(__HIP__)
    "--offload-arch=" + arch,
#else
    "-arch=" + arch,
#endif
#if defined(__HIP_PLATFORM_AMD__)
    // HIP does not enable warp sync functions by default (yet) in ROCm 6.2
    "-DHIP_ENABLE_WARP_SYNC_BUILTINS",
#endif
    "-I" + cuda_include_path,
    "-DN=" + std::to_string(N_) + "UL",
    "-DWARP_SIZE=" + std::to_string(warp_size)
  };

  if (input_complex_axis_location_ == ComplexAxisLocation::complex_planar) {
    options.push_back("-DINPUT_COMPLEX_PLANAR");
  } else if (input_complex_axis_location_ ==
             ComplexAxisLocation::complex_interleaved) {
    options.push_back("-DINPUT_COMPLEX_INTERLEAVED");
  }

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
  if (direction_ == Direction::forward) {
    function_ = std::make_unique<cu::Function>(*module_, "pack_bits");
  } else if (direction_ == Direction::backward) {
    function_ = std::make_unique<cu::Function>(*module_, "unpack_bits");
  } else {
    throw std::runtime_error("Unknown packing direction");
  }
}

Packing::Packing(size_t N, Direction direction, cu::Device &device,
                 cu::Stream &stream,
                 ComplexAxisLocation input_complex_axis_location)
    : impl_(std::make_unique<Impl>(N, direction, device, stream,
                                   input_complex_axis_location)){};

Packing::Packing(size_t N, Direction direction, CUdevice &device,
                 CUstream &stream,
                 ComplexAxisLocation input_complex_axis_location)
    : N_(N), device_(new cu::Device(device)), stream_(new cu::Stream(stream)) {
  impl_ = std::make_unique<Impl>(N, direction, *device_, *stream_,
                                 input_complex_axis_location);
}

Packing::~Packing() = default;

void Packing::Run(cu::HostMemory &h_input, cu::DeviceMemory &d_output) {
  impl_->Run(h_input, d_output);
}

void Packing::Run(unsigned char *h_input, CUdeviceptr d_output) {
  cu::HostMemory h_input_(h_input, N_);
  cu::DeviceMemory d_output_(d_output);
  impl_->Run(h_input_, d_output_);
}

void Packing::Run(cu::DeviceMemory &d_input, cu::DeviceMemory &d_output) {
  impl_->Run(d_input, d_output);
}

void Packing::Run(CUdeviceptr d_input, CUdeviceptr d_output) {
  cu::DeviceMemory d_input_(d_input);
  cu::DeviceMemory d_output_(d_output);
  impl_->Run(d_input_, d_output_);
}

} // end namespace ccglib::packing
