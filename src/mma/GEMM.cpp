#include <cassert>
#include <cmath>
#include <iostream>

#include <cudawrappers/nvrtc.hpp>

#include <ccglib/gemm/mma.h>
#include <ccglib/helper.h>
#include <ccglib/precision.h>

#include "Kernel.h"

namespace ccglib::mma {

class GEMM::Impl {
public:
  Impl(size_t B_, size_t M_, size_t N_, size_t K_, cu::Device &device_,
       cu::Stream &stream_, Precision precision, Variant variant,
       ComplexAxisLocation c_complex_axis_location, MemOrder c_mem_order,
       MemOrder a_mem_order, MemOrder b_mem_order);

  void Run(cu::DeviceMemory &d_a, cu::DeviceMemory &d_b, cu::DeviceMemory &d_c);

private:
  void compile_kernel();

  ComplexAxisLocation c_complex_axis_location_;
  MemOrder a_mem_order_;
  MemOrder b_mem_order_;
  MemOrder c_mem_order_;
  Variant variant_;
  Kernel kernel_;

  size_t B_;
  size_t K_;
  size_t M_;
  size_t N_;

  dim3 threads_;
  dim3 grid_;

  cu::Device &device_;
  cu::Stream &stream_;

  std::unique_ptr<cu::Module> module_;
  std::unique_ptr<cu::Function> function_;
  std::unique_ptr<cu::Function> downcast_function_;
};

GEMM::Impl::Impl(size_t B_, size_t M_, size_t N_, size_t K_,
                 cu::Device &device_, cu::Stream &stream_, Precision precision,
                 Variant variant, ComplexAxisLocation c_complex_axis_location,
                 MemOrder a_mem_order, MemOrder b_mem_order,
                 MemOrder c_mem_order)
    : B_(B_), M_(M_), N_(N_), K_(K_), device_(device_), stream_(stream_),
      c_complex_axis_location_(c_complex_axis_location), variant_(variant),
      c_mem_order_(c_mem_order), a_mem_order_(a_mem_order),
      b_mem_order_(b_mem_order), kernel_(Kernel(precision, variant)) {
  const Kernel::Parameters parameters = kernel_.GetParameters();
  threads_ = kernel_.GetThreads(device_);
  grid_ = dim3(ccglib::helper::ceildiv(N_, parameters.n_per_block),
               ccglib::helper::ceildiv(M_, parameters.m_per_block), B_);

  const bool precision_is_int1 = (precision.input_type == ValueType::int1);
  const bool variant_is_basic = variant == Variant::basic;
  const bool c_complex_axis_is_last =
      c_complex_axis_location == ComplexAxisLocation::complex_last;

  if (precision_is_int1 && c_complex_axis_is_last) {
    throw std::runtime_error(
        "complex-last output is not supported in int1 precision");
  }

  if (variant_is_basic && c_complex_axis_is_last) {
    throw std::runtime_error(
        "complex-last output is not supported in basic variant");
  }

#if defined(DEBUG)
  std::cout << "Problem size (B, M, N, K): (" << B_ << ", " << M_ << ", " << N_
            << ", " << K_ << ")" << std::endl;
  std::cout << "Block sizes (m, n): (" << parameters.m_per_block << ", "
            << parameters.n_per_block << ")" << std::endl;
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
  const ccglib::Precision precision = kernel_.GetPrecision();

  std::vector<const void *> parameters = {d_c.parameter(), d_a.parameter(),
                                          d_b.parameter()};

  // A input type of float32 and output type of float16 requires special
  // handling in order to downcast the 32 bit output to the 16 bit output.
  if (precision.input_type == ValueType::float32 &&
      precision.output_type == ValueType::float16) {
    // If the downcast function has not been compiled, throw an error.
    // This should never happen since the if-condition above ensures that
    // the right conditions are met.
    if (!downcast_function_) {
      throw std::runtime_error("Unexpected error: downcast function is null");
    }

    // Create an temporary output buffer that is large enough to hold the 32
    // bit output of the kernel. Then downcast the 32 bit output to the 16 bit
    // output 'd_c'.
    const size_t temp_buffer_size =
        (d_c.size() + (precision.GetOutputBits() / CHAR_BIT) - 1) /
        (precision.GetOutputBits() / CHAR_BIT) *
        (precision.GetInputBits() / CHAR_BIT);

    // Allocate the temporary buffer
    cu::DeviceMemory d_temp = stream_.memAllocAsync(temp_buffer_size);

    stream_.launchKernel(
        *function_, grid_.x, grid_.y, grid_.z, threads_.x, threads_.y,
        threads_.z, 0, {d_temp.parameter(), d_a.parameter(), d_b.parameter()});

    // Determine the execution layout.
    const size_t d_c_count =
        d_c.size() / (precision.GetOutputBits() / CHAR_BIT);
    const size_t threads = 256;
    const size_t blocks = (d_c_count + threads - 1) / threads;

    std::vector<const void *> downcast_parameters = {
        {d_temp.parameter(), d_c.parameter(), &temp_buffer_size}};

    // Copy the output from the temporary buffer to d_c.
    stream_.launchKernel(*downcast_function_, threads, 1, 1, blocks, 1, 1, 0,
                         downcast_parameters);

    // Free the temporary buffer after it has been copied to d_c.
    stream_.memFreeAsync(d_temp);
  } else {
    stream_.launchKernel(*function_, grid_.x, grid_.y, grid_.z, threads_.x,
                         threads_.y, threads_.z, 0, parameters);
  }
}

void GEMM::Impl::compile_kernel() {
  const std::string cuda_include_path = nvrtc::findIncludePath();

  const std::string arch = device_.getArch();
  const unsigned warp_size =
      device_.getAttribute(CU_DEVICE_ATTRIBUTE_WARP_SIZE);

  const Kernel::Parameters parameters = kernel_.GetParameters();

  std::vector<std::string> options = {
    "-std=c++17",
#if defined(__HIP__)
    "--offload-arch=" + arch,
#else
    "-arch=" + arch,
#endif
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
    "-DNBIT_IN=" + std::to_string(kernel_.GetPrecision().GetInputBits()),
    "-DNBIT_OUT=" + std::to_string(kernel_.GetPrecision().GetOutputBits()),
    "-DWARP_SIZE=" + std::to_string(warp_size),
    "-DM_PER_BLOCK=" + std::to_string(parameters.m_per_block),
    "-DM_PER_WARP=" + std::to_string(parameters.m_per_warp),
    "-DM_PER_WMMA=" + std::to_string(parameters.m_per_wmma),
    "-DN_PER_BLOCK=" + std::to_string(parameters.n_per_block),
    "-DN_PER_WARP=" + std::to_string(parameters.n_per_warp),
    "-DN_PER_WMMA=" + std::to_string(parameters.n_per_wmma),
    "-DK_PER_WMMA=" + std::to_string(parameters.k_per_wmma),
    "-DNBUFFER=" + std::to_string(parameters.nbuffer)
  };

  if (c_complex_axis_location_ == ComplexAxisLocation::complex_middle) {
    options.push_back("-DC_COMPLEX_MIDDLE");
  } else if (c_complex_axis_location_ == ComplexAxisLocation::complex_last) {
    options.push_back("-DC_COMPLEX_LAST");
  }

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

  try {
    downcast_function_ =
        std::make_unique<cu::Function>(*module_, "downcast_kernel");
  } catch (cu::Error &) {
    // Downcast kernel is not implemented for the int1 kernel, so don't throw.
    downcast_function_ = nullptr;
  }
}

GEMM::GEMM(const size_t B_, const size_t M_, const size_t N_, const size_t K_,
           const size_t, cu::Device &device_, cu::Stream &stream_,
           const Precision precision, const Variant variant,
           const ComplexAxisLocation c_complex_axis_location,
           const MemOrder c_mem_order, const MemOrder a_mem_order,
           const MemOrder b_mem_order)
    : impl_(std::make_unique<Impl>(B_, M_, N_, K_, device_, stream_, precision,
                                   variant, c_complex_axis_location,
                                   a_mem_order, b_mem_order, c_mem_order)){};
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
