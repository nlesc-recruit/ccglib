#include <limits>

#include <cudawrappers/cu.hpp>

#include <ccglib/common/helper.h>
#include <ccglib/gemm/mma.h>
#include <ccglib/packing/packing.h>
#include <ccglib/pipeline/pipeline.h>
#include <ccglib/transpose/transpose.h>

namespace ccglib::pipeline {

class Pipeline::Impl {
public:
  Impl(size_t B, size_t M, size_t N, size_t K, cu::Device &device,
       cu::Stream &stream, ComplexAxisLocation input_complex_axis_location,
       ComplexAxisLocation output_complex_axis_location,
       mma::MemOrder a_mem_order, mma::MemOrder b_mem_order,
       mma::MemOrder c_mem_order, ValuePrecision input_precision,
       ValuePrecision output_precision, mma::Variant variant);

  void Run(cu::DeviceMemory &d_a, cu::DeviceMemory &d_b, cu::DeviceMemory &d_c);

private:
  static const size_t kComplex = 2;
  const bool needs_packing_;
  const bool needs_transpose_;

  cu::Device device_;
  cu::Stream stream_;

  std::unique_ptr<cu::DeviceMemory> d_a_packed_;
  std::unique_ptr<cu::DeviceMemory> d_b_packed_;
  std::unique_ptr<cu::DeviceMemory> d_a_trans_;
  std::unique_ptr<cu::DeviceMemory> d_b_trans_;

  std::unique_ptr<ccglib::packing::Packing> packing_a_;
  std::unique_ptr<ccglib::packing::Packing> packing_b_;
  std::unique_ptr<ccglib::transpose::Transpose> transpose_a_;
  std::unique_ptr<ccglib::transpose::Transpose> transpose_b_;
  std::unique_ptr<ccglib::mma::GEMM> gemm_;
};

Pipeline::Impl::Impl(size_t B, size_t M, size_t N, size_t K, cu::Device &device,
                     cu::Stream &stream,
                     ComplexAxisLocation input_complex_axis_location,
                     ComplexAxisLocation output_complex_axis_location,
                     mma::MemOrder a_mem_order, mma::MemOrder b_mem_order,
                     mma::MemOrder c_mem_order, ValuePrecision input_precision,
                     ValuePrecision output_precision, mma::Variant variant)
    : needs_packing_(input_precision == ccglib::int1),
      needs_transpose_(variant == ccglib::mma::opt), device_(device),
      stream_(stream) {

  if (needs_packing_) {
    const size_t num_a = B * kComplex * M * K;
    const size_t num_b = B * kComplex * N * K;
    packing_a_ =
        std::make_unique<ccglib::packing::Packing>(num_a, device_, stream_);
    packing_b_ = std::make_unique<ccglib::packing::Packing>(
        num_b * kComplex * N * K, device_, stream_);

    // packing output is of int32 type, so round up input size to multiple of
    // int32 size.
    const size_t bytes_packed_a =
        sizeof(int) * ccglib::helper::ceildiv(num_a / CHAR_BIT, sizeof(int));
    const size_t bytes_packed_b =
        sizeof(int) * ccglib::helper::ceildiv(num_b / CHAR_BIT, sizeof(int));
    d_a_packed_ = std::make_unique<cu::DeviceMemory>(bytes_packed_a);
    d_b_packed_ = std::make_unique<cu::DeviceMemory>(bytes_packed_b);
  }

  const Precision precision(input_precision, output_precision);
  // x, y, z = m_per_block, n_per_block, k_per_wmma
  const dim3 dimensions = ccglib::mma::GEMM::GetDimensions(precision, variant);

  if (needs_transpose_) {
    transpose_a_ = std::make_unique<ccglib::transpose::Transpose>(
        B, M, K, dimensions.x, dimensions.z, precision.GetInputBits(), device_,
        stream_, input_complex_axis_location);
    transpose_b_ = std::make_unique<ccglib::transpose::Transpose>(
        B, N, K, dimensions.y, dimensions.z, precision.GetOutputBits(), device_,
        stream_, input_complex_axis_location);

    // transposed size may be bigger than input due to padding on block level
    const size_t m_padded =
        dimensions.x * ccglib::helper::ceildiv(M, dimensions.x);
    const size_t n_padded =
        dimensions.y * ccglib::helper::ceildiv(N, dimensions.y);
    const size_t k_padded =
        dimensions.z * ccglib::helper::ceildiv(K, dimensions.y);
    const size_t bytes_trans_a = B * kComplex * m_padded * k_padded;
    const size_t bytes_trans_b = B * kComplex * n_padded * k_padded;

    d_a_trans_ = std::make_unique<cu::DeviceMemory>(bytes_trans_a);
    d_b_trans_ = std::make_unique<cu::DeviceMemory>(bytes_trans_b);
  }

  gemm_ = std::make_unique<ccglib::mma::GEMM>(
      B, M, N, K, device_, stream_, precision, variant,
      output_complex_axis_location, c_mem_order, a_mem_order, b_mem_order);
}

void Pipeline::Impl::Run(cu::DeviceMemory &d_a, cu::DeviceMemory &d_b,
                         cu::DeviceMemory &d_c) {}

Pipeline::Pipeline(size_t B, size_t M, size_t N, size_t K, cu::Device &device,
                   cu::Stream &stream,
                   ComplexAxisLocation input_complex_axis_location,
                   ComplexAxisLocation output_complex_axis_location,
                   mma::MemOrder a_mem_order, mma::MemOrder b_mem_order,
                   mma::MemOrder c_mem_order, ValuePrecision input_precision,
                   ValuePrecision output_precision, mma::Variant variant)
    : device_(std::make_unique<cu::Device>(device)),
      stream_(std::make_unique<cu::Stream>(stream)) {
  impl_ = std::make_unique<Impl>(
      B, M, N, K, *device_, *stream_, input_complex_axis_location,
      output_complex_axis_location, a_mem_order, b_mem_order, c_mem_order,
      input_precision, output_precision, variant);
}

void Pipeline::Run(cu::HostMemory &a, cu::HostMemory &b, cu::HostMemory &c) {
  cu::DeviceMemory d_a(a.size());
  cu::DeviceMemory d_b(b.size());
  cu::DeviceMemory d_c(c.size());
  stream_->memcpyHtoDAsync(d_a, a, a.size());
  stream_->memcpyHtoDAsync(d_b, b, b.size());
  impl_->Run(d_a, d_b, d_c);
  stream_->memcpyDtoHAsync(c, d_c, c.size());
  stream_->synchronize();
}

void Pipeline::Run(cu::DeviceMemory &d_a, cu::DeviceMemory &d_b,
                   cu::DeviceMemory &d_c) {
  impl_->Run(d_a, d_b, d_c);
}

} // namespace ccglib::pipeline
