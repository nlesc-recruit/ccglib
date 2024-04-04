#include <cassert>
#include <iostream>
#include <numeric>

#include <cudawrappers/nvrtc.hpp>

#include "GEMM.h"
#include "config.h"
#include "helper/helper.h"

extern const char _binary_kernels_gemm_kernel_cu_start,
    _binary_kernels_gemm_kernel_cu_end;

namespace ccglib::mma {

GEMM::GEMM(size_t beams_, size_t samples_, size_t frames_,
           size_t nr_input_bits_, size_t nr_output_bits, cu::Device &device_,
           cu::Stream &stream_, Variant variant)
    : beams_(beams_), samples_(samples_), frames_(frames_),
      nr_input_bits_(nr_input_bits_), device_(device_), stream_(stream_),
      variant_(variant) {
  assert(kNrOutputBits == kNrOutputBits);
  threads_ = dim3(kWarpSize, kFramesPerBlock / kFramesPerWarp,
                  kBeamsPerBlock / kBeamsPerWarp);
  grid_ = dim3(helper::ceildiv(frames_, kFramesPerBlock),
               helper::ceildiv(beams_, kBeamsPerBlock));

#if defined(DEBUG)
  std::cout << "Problem size (M, N, K): (" << beams_ << ", " << frames_ << ", "
            << samples_ << ")" << std::endl;
  std::cout << "Thread block size: (" << threads_.x << ", " << threads_.y
            << ", " << threads_.z << ")" << std::endl;
  std::cout << "Threads per block: " << threads_.x * threads_.y * threads_.z
            << std::endl;
  std::cout << "Grid size: (" << grid_.x << ", " << grid_.y << ", " << grid_.z
            << ")" << std::endl;
#endif

  compile_kernel();
}

inline const char *to_string(ccglib::mma::GEMM::Variant v) {
  switch (v) {
  case ccglib::mma::GEMM::basic:
    return "wmma_complex_gemm_basic";
  case ccglib::mma::GEMM::opt:
    return "wmma_complex_gemm_opt";
  default:
    return "";
  }
}

void GEMM::compile_kernel() {
  const std::string cuda_include_path =
      std::string(getenv("CUDA_HOME")) + "/include";
  const std::string lib_include_path = std::string(INSTALL_INCLUDE_DIR);

  const int capability = helper::get_capability(device_);

  std::vector<std::string> options = {
      "-std=c++17",
      "-arch=sm_" + std::to_string(capability),
      "-I" + cuda_include_path,
      "-I" + lib_include_path,
      "-Dblock_size_x=" + std::to_string(threads_.x),
      "-Dblock_size_y=" + std::to_string(threads_.y),
      "-Dblock_size_z=" + std::to_string(threads_.z),
      "-DM=" + std::to_string(beams_),
      "-D_N=" + std::to_string(frames_),
      "-DK=" + std::to_string(samples_),
      "-DNBIT=" + std::to_string(nr_input_bits_),
      "-DM_PER_BLOCK=" + std::to_string(kBeamsPerBlock),
      "-DM_PER_WARP=" + std::to_string(kBeamsPerWarp),
      "-DM_PER_WMMA=" + std::to_string(kBeamsPerWMMA),
      "-DN_PER_BLOCK=" + std::to_string(kFramesPerBlock),
      "-DN_PER_WARP=" + std::to_string(kFramesPerWarp),
      "-DN_PER_WMMA=" + std::to_string(kFramesPerWMMA),
      "-DK_PER_WMMA=" + std::to_string(kSamplesPerWMMA),
      "-DWARP_SIZE=" + std::to_string(kWarpSize),
      "-DNBUFFER=" + std::to_string(kNBuffer)};

  const std::string kernel(&_binary_kernels_gemm_kernel_cu_start,
                           &_binary_kernels_gemm_kernel_cu_end);

  nvrtc::Program program(kernel, "gemm_kernel.cu");

  try {
    program.compile(options);
  } catch (nvrtc::Error &error) {
    std::cerr << program.getLog();
    throw;
  }

  module_ = std::make_unique<cu::Module>(
      static_cast<const void *>(program.getPTX().data()));
  function_ = std::make_unique<cu::Function>(*module_, to_string(variant_));
}

void GEMM::run(cu::DeviceMemory &d_a, cu::DeviceMemory &d_b,
               cu::DeviceMemory &d_c) {

  std::vector<const void *> parameters = {d_c.parameter(), d_a.parameter(),
                                          d_b.parameter()};

  stream_.launchKernel(*function_, grid_.x, grid_.y, grid_.z, threads_.x,
                       threads_.y, threads_.z, 0, parameters);
}

} // end namespace ccglib::mma