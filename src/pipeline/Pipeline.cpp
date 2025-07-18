#include <cudawrappers/cu.hpp>
#include <limits>

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
  const bool requires_packing_;
  const bool requires_transpose_;
  const ComplexAxisLocation input_complex_axis_location_;

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
    : requires_packing_(input_precision == ccglib::int1),
      requires_transpose_(variant == ccglib::mma::opt), device_(device),
      stream_(stream),
      input_complex_axis_location_(input_complex_axis_location) {

  if (requires_packing_) {
    const size_t num_a = B * kComplex * M * K;
    const size_t num_b = B * kComplex * N * K;
    packing_a_ =
        std::make_unique<ccglib::packing::Packing>(num_a, device_, stream_);
    packing_b_ =
        std::make_unique<ccglib::packing::Packing>(num_b, device_, stream_);

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

  if (requires_transpose_) {
    transpose_a_ = std::make_unique<ccglib::transpose::Transpose>(
        B, M, K, dimensions.x, dimensions.z, precision.GetInputBits(), device_,
        stream_, input_complex_axis_location_);
    transpose_b_ = std::make_unique<ccglib::transpose::Transpose>(
        B, N, K, dimensions.y, dimensions.z, precision.GetInputBits(), device_,
        stream_, input_complex_axis_location_);

    // transposed size may be bigger than input due to padding on block level
    const size_t m_padded =
        dimensions.x * ccglib::helper::ceildiv(M, dimensions.x);
    const size_t n_padded =
        dimensions.y * ccglib::helper::ceildiv(N, dimensions.y);
    const size_t k_padded =
        dimensions.z * ccglib::helper::ceildiv(K, dimensions.z);
    const size_t bytes_trans_a = B * kComplex * m_padded * k_padded *
                                 precision.GetInputBits() / CHAR_BIT;
    const size_t bytes_trans_b = B * kComplex * n_padded * k_padded *
                                 precision.GetInputBits() / CHAR_BIT;

    d_a_trans_ = std::make_unique<cu::DeviceMemory>(bytes_trans_a);
    d_b_trans_ = std::make_unique<cu::DeviceMemory>(bytes_trans_b);
  }

  gemm_ = std::make_unique<ccglib::mma::GEMM>(
      B, M, N, K, device_, stream_, precision, variant,
      output_complex_axis_location, c_mem_order, a_mem_order, b_mem_order);
}

void Pipeline::Impl::Run(cu::DeviceMemory &d_a, cu::DeviceMemory &d_b,
                         cu::DeviceMemory &d_c) {

  if (requires_packing_) {
    packing_a_->Run(d_a, *d_a_packed_, ccglib::forward,
                    input_complex_axis_location_);
    packing_b_->Run(d_b, *d_b_packed_, ccglib::forward,
                    input_complex_axis_location_);
  }

  if (requires_transpose_) {
    cu::DeviceMemory &input_a = requires_packing_ ? *d_a_packed_ : d_a;
    cu::DeviceMemory &input_b = requires_packing_ ? *d_b_packed_ : d_b;
    transpose_a_->Run(input_a, *d_a_trans_);
    transpose_b_->Run(input_b, *d_b_trans_);
  }

  cu::DeviceMemory &input_a = requires_transpose_ ? *d_a_trans_ : d_a;
  cu::DeviceMemory &input_b = requires_transpose_ ? *d_b_trans_ : d_b;
  gemm_->Run(input_a, input_b, d_c);
}

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

Pipeline::Pipeline(size_t B, size_t M, size_t N, size_t K, CUdevice &device,
                   CUstream &stream,
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

void Pipeline::Run(CUdeviceptr d_a, CUdeviceptr d_b, CUdeviceptr d_c) {
  cu::DeviceMemory d_a_(d_a);
  cu::DeviceMemory d_b_(d_b);
  cu::DeviceMemory d_c_(d_c);
  impl_->Run(d_a_, d_b_, d_c_);
}

Pipeline::~Pipeline() = default;

} // namespace ccglib::pipeline
