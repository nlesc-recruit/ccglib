#include <iostream>

#include <cudawrappers/nvrtc.hpp>

#include <ccglib/gemm/mma.h>
#include <ccglib/helper.h>

#include "Kernel.h"

namespace ccglib::mma {

class GEMM::Impl {
public:
  Impl(size_t B_, size_t M_, size_t N_, size_t K_, size_t nr_input_bits_,
       cu::Device &device_, cu::Stream &stream_, Precision precision,
       Variant variant, MemOrder c_mem_order, MemOrder a_mem_order,
       MemOrder b_mem_order);

  void Run(cu::DeviceMemory &d_a, cu::DeviceMemory &d_b, cu::DeviceMemory &d_c);

private:
  void compile_kernel();

  MemOrder a_mem_order_;
  MemOrder b_mem_order_;
  MemOrder c_mem_order_;
  Variant variant_;
  const Kernel &kernel_;

  size_t B_;
  size_t K_;
  size_t M_;
  size_t N_;

  size_t nr_input_bits_;

  dim3 threads_;
  dim3 grid_;

  cu::Device &device_;
  cu::Stream &stream_;

  std::unique_ptr<cu::Module> module_;
  std::unique_ptr<cu::Function> function_;
};

GEMM::Impl::Impl(size_t B_, size_t M_, size_t N_, size_t K_,
                 size_t nr_input_bits_, cu::Device &device_,
                 cu::Stream &stream_, Precision precision, Variant variant,
                 MemOrder a_mem_order, MemOrder b_mem_order,
                 MemOrder c_mem_order)
    : B_(B_), M_(M_), N_(N_), K_(K_), nr_input_bits_(nr_input_bits_),
      device_(device_), stream_(stream_), variant_(variant),
      c_mem_order_(c_mem_order), a_mem_order_(a_mem_order),
      b_mem_order_(b_mem_order), kernel_(Kernel(precision, variant)) {
  const Kernel::Parameters parameters = kernel_.GetParameters();
  threads_ = kernel_.GetThreads();
  grid_ = dim3(ccglib::helper::ceildiv(N_, parameters.n_per_block),
               ccglib::helper::ceildiv(M_, parameters.m_per_block), B_);

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

void GEMM::Impl::Run(cu::DeviceMemory &d_a, cu::DeviceMemory &d_b,
                     cu::DeviceMemory &d_c) {

  std::vector<const void *> parameters = {d_c.parameter(), d_a.parameter(),
                                          d_b.parameter()};

  stream_.launchKernel(*function_, grid_.x, grid_.y, grid_.z, threads_.x,
                       threads_.y, threads_.z, 0, parameters);
}

void GEMM::Impl::compile_kernel() {
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
      "-DBATCH_SIZE=" + std::to_string(B_) + "UL",
      "-DM_GLOBAL=" + std::to_string(M_) + "UL",
      "-DN_GLOBAL=" + std::to_string(N_) + "UL",
      "-DK_GLOBAL=" + std::to_string(K_) + "UL",
      "-DK_PADDING=" + std::to_string(0) +
          "UL", // will be required when K is not a multiple of K_PER_WMMA
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

  if (a_mem_order_ == MemOrder::row_major) {
    options.push_back("-DA_ROW_MAJOR");
  } else {
    options.push_back("-DA_COL_MAJOR");
  }

  if (b_mem_order_ == MemOrder::row_major) {
    options.push_back("-DB_ROW_MAJOR");
  } else {
    options.push_back("-DB_COL_MAJOR");
  }

  if (c_mem_order_ == MemOrder::row_major) {
    options.push_back("-DC_ROW_MAJOR");
  } else {
    options.push_back("-DC_COL_MAJOR");
  }

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

GEMM::GEMM(size_t B_, size_t M_, size_t N_, size_t K_, size_t nr_input_bits_,
           cu::Device &device_, cu::Stream &stream_, Precision precision,
           Variant variant, MemOrder c_mem_order, MemOrder a_mem_order,
           MemOrder b_mem_order)
    : impl_(std::make_unique<Impl>(B_, M_, N_, K_, nr_input_bits_, device_,
                                   stream_, precision, variant, a_mem_order,
                                   b_mem_order, c_mem_order)){};

GEMM::~GEMM() = default;

void GEMM::Run(cu::DeviceMemory &d_a, cu::DeviceMemory &d_b,
               cu::DeviceMemory &d_c) {
  impl_->Run(d_a, d_b, d_c);
}

dim3 GEMM::GetDimensions(Precision precision, Variant variant) {
  const Kernel kernel(precision, variant);
  const Kernel::Parameters parameters = kernel.GetParameters();
  return dim3(parameters.m_per_block, parameters.n_per_block,
              parameters.k_per_wmma);
}

} // end namespace ccglib::mma
