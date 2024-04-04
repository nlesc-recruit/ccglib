#include <iostream>
#include <numeric>

#include <cudawrappers/nvrtc.hpp>

#include "GEMM.h"
#include "config.h"
#include "helper/helper.h"

extern const char _binary_kernels_gemm_kernel_cu_start,
    _binary_kernels_gemm_kernel_cu_end;

namespace ccglib::mma {

GEMM::GEMM(size_t beams, size_t samples, size_t frames, size_t nr_input_bits,
           size_t nr_output_bits, cu::Device &device, cu::Stream &stream)
    : beams(beams), samples(samples), frames(frames),
      nr_input_bits(nr_input_bits), nr_output_bits(nr_output_bits),
      device(device), stream(stream) {
  threads = dim3(warp_size, kFramesPerBlock / frames_per_warp,
                 kBeamsPerBlock / beams_per_warp);
  grid = dim3(helper::ceildiv(frames, kFramesPerBlock),
              helper::ceildiv(beams, kBeamsPerBlock));

#if defined(DEBUG)
  std::cout << "Problem size (M, N, K): (" << beams << ", " << frames << ", "
            << samples << ")" << std::endl;
  std::cout << "Thread block size: (" << threads.x << ", " << threads.y << ", "
            << threads.z << ")" << std::endl;
  std::cout << "Threads per block: " << threads.x * threads.y * threads.z
            << std::endl;
  std::cout << "Grid size: (" << grid.x << ", " << grid.y << ", " << grid.z
            << ")" << std::endl;
#endif

  compile_kernel();
}

void GEMM::compile_kernel() {
  const std::string cuda_include_path =
      std::string(getenv("CUDA_HOME")) + "/include";
  const std::string lib_include_path = std::string(INSTALL_INCLUDE_DIR);

  const int capability = helper::get_capability(device);

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
      "-DNBIT=" + std::to_string(nr_input_bits),
      "-DM_PER_BLOCK=" + std::to_string(kBeamsPerBlock),
      "-DM_PER_WARP=" + std::to_string(beams_per_warp),
      "-DM_PER_WMMA=" + std::to_string(beams_per_wmma),
      "-DN_PER_BLOCK=" + std::to_string(kFramesPerBlock),
      "-DN_PER_WARP=" + std::to_string(frames_per_warp),
      "-DN_PER_WMMA=" + std::to_string(frames_per_wmma),
      "-DK_PER_WMMA=" + std::to_string(kSamplesPerWMMA),
      "-DWARP_SIZE=" + std::to_string(warp_size),
      "-DNBUFFER=" + std::to_string(nbuffer)};

  const std::string kernel(&_binary_kernels_gemm_kernel_cu_start,
                           &_binary_kernels_gemm_kernel_cu_end);

  nvrtc::Program program(kernel, "gemm_kernel.cu");

  try {
    program.compile(options);
  } catch (nvrtc::Error &error) {
    std::cerr << program.getLog();
    throw;
  }

  module = std::make_unique<cu::Module>(
      static_cast<const void *>(program.getPTX().data()));
  function = std::make_unique<cu::Function>(*module, "wmma_complex_gemm_opt");
}

void GEMM::run(cu::DeviceMemory &d_a, cu::DeviceMemory &d_b,
               cu::DeviceMemory &d_c) {

  std::vector<const void *> parameters = {d_c.parameter(), d_a.parameter(),
                                          d_b.parameter()};

  stream.launchKernel(*function, grid.x, grid.y, grid.z, threads.x, threads.y,
                      threads.z, 0, parameters);
}

} // end namespace ccglib::mma