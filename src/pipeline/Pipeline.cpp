#include <cudawrappers/cu.hpp>

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
       ValuePrecision output_precision, mma::Variant variant) {}

  void Run(cu::DeviceMemory &d_a, cu::DeviceMemory &d_b,
           cu::DeviceMemory &d_c) {}
};

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

} // namespace ccglib::pipeline
