#ifndef PIPELINE_H_
#define PIPELINE_H_

#include <memory>

#include <ccglib/common/complex_order.h>
#include <ccglib/common/value_precision.h>
#include <ccglib/gemm/mem_order.h>
#include <ccglib/gemm/variant.h>

// Forward declaration of cudawrappers types
namespace cu {
class Device;
class HostMemory;
class Stream;
} // namespace cu

namespace ccglib::pipeline {

class Pipeline {
public:
  Pipeline(size_t B, size_t M, size_t N, size_t K, cu::Device &device,
           cu::Stream &stream, ComplexAxisLocation input_complex_axis_location,
           ComplexAxisLocation output_complex_axis_location,
           mma::MemOrder a_mem_order, mma::MemOrder b_mem_order,
           mma::MemOrder c_mem_order, ValuePrecision input_precision,
           ValuePrecision output_precision,
           mma::Variant variant = mma::Variant::opt);
  ~Pipeline();
  void Run(cu::HostMemory &a, cu::HostMemory &b, cu::HostMemory &c);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
  std::unique_ptr<cu::Device> device_;
  std::unique_ptr<cu::Stream> stream_;
};

} // namespace ccglib::pipeline

#endif // PIPELINE_H_
