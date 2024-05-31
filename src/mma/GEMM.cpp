#include <cassert>
#include <iostream>
#include <numeric>

#include <cudawrappers/nvrtc.hpp>

#include <ccglib/gemm/mma.h>
#include <ccglib/helper.h>

namespace ccglib::mma {

GEMM::GEMM(size_t B_, size_t M_, size_t K_, size_t N_, size_t nr_input_bits_,
           cu::Device &device_, cu::Stream &stream_, Precision precision,
           Variant variant)
    : B_(B_), M_(M_), K_(K_), N_(N_), nr_input_bits_(nr_input_bits_),
      device_(device_), stream_(stream_), variant_(variant),
      kernel_(Kernel(precision, variant)) {
  const Kernel::Parameters parameters = kernel_.GetParameters();
  threads_ = kernel_.GetThreads();
  kMPerBlock = parameters.m_per_block;
  kNPerBlock = parameters.n_per_block;
  kKPerWMMA = parameters.k_per_wmma;
  grid_ = dim3(helper::ceildiv(N_, parameters.n_per_block),
               helper::ceildiv(M_, parameters.m_per_block), B_);

#if defined(DEBUG)
  std::cout << "Problem size (B, M, N, K): (" << B_ << ", " << M_ << ", " << N_
            << ", " << K_ << ")" << std::endl;
  std::cout << "Thread block size: (" << threads_.x << ", " << threads_.y
            << ", " << threads_.z << ")" << std::endl;
  std::cout << "Threads per block: " << threads_.x * threads_.y * threads_.z
            << std::endl;
  std::cout << "Grid size: (" << grid_.x << ", " << grid_.y << ", " << grid_.z
            << ")" << std::endl;
#endif
  compile_kernel();
}

void GEMM::compile_kernel() {
  const std::string cuda_include_path = nvrtc::findIncludePath();

  const int capability = helper::get_capability(device_);

  const Kernel::Parameters parameters = kernel_.GetParameters();

  std::vector<std::string> options = {
      "-std=c++17",
      "-arch=sm_" + std::to_string(capability),
      "-I" + cuda_include_path,
      "-Dblock_size_x=" + std::to_string(threads_.x),
      "-Dblock_size_y=" + std::to_string(threads_.y),
      "-Dblock_size_z=" + std::to_string(threads_.z),
      "-DBATCH_SIZE=" + std::to_string(B_),
      "-DM_GLOBAL=" + std::to_string(M_),
      "-DN_GLOBAL=" + std::to_string(N_),
      "-DK_GLOBAL=" + std::to_string(K_),
      "-DK_PADDING=" +
          std::to_string(
              0), // will be required when K is not a multiple of K_PER_WMMA
      "-DNBIT=" + std::to_string(nr_input_bits_),
      "-DM_PER_BLOCK=" + std::to_string(parameters.m_per_block),
      "-DM_PER_WARP=" + std::to_string(parameters.m_per_warp),
      "-DM_PER_WMMA=" + std::to_string(parameters.m_per_wmma),
      "-DN_PER_BLOCK=" + std::to_string(parameters.n_per_block),
      "-DN_PER_WARP=" + std::to_string(parameters.n_per_warp),
      "-DN_PER_WMMA=" + std::to_string(parameters.n_per_wmma),
      "-DK_PER_WMMA=" + std::to_string(parameters.k_per_wmma),
      "-DWARP_SIZE=" + std::to_string(parameters.warp_size),
      "-DNBUFFER=" + std::to_string(parameters.nbuffer)};

  nvrtc::Program program(kernel_.GetSource(), "gemm_kernel.cu");

  try {
    program.compile(options);
  } catch (nvrtc::Error &error) {
    std::cerr << program.getLog();
    throw;
  }

  module_ = std::make_unique<cu::Module>(
      static_cast<const void *>(program.getPTX().data()));
  function_ =
      std::make_unique<cu::Function>(*module_, kernel_.GetName().c_str());
}

void GEMM::run(cu::DeviceMemory &d_a, cu::DeviceMemory &d_b,
               cu::DeviceMemory &d_c) {

  std::vector<const void *> parameters = {d_c.parameter(), d_a.parameter(),
                                          d_b.parameter()};

  stream_.launchKernel(*function_, grid_.x, grid_.y, grid_.z, threads_.x,
                       threads_.y, threads_.z, 0, parameters);
}

} // end namespace ccglib::mma
