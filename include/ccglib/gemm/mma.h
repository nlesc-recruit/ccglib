#ifndef MMA_GEMM_H_
#define MMA_GEMM_H_

#include <ccglib/gemm/complex_order.h>
#include <ccglib/gemm/mem_order.h>
#include <ccglib/gemm/variant.h>
#include <ccglib/precision.h>
#include <cudawrappers/cu.hpp>

namespace ccglib::mma {

class GEMM {
public:
  GEMM(size_t B_, size_t M_, size_t N_, size_t K_, size_t, cu::Device &device_,
       cu::Stream &stream_, Precision precision, Variant Variant = Variant::opt,
       ComplexAxisLocation c_complex_axis_location =
           ComplexAxisLocation::complex_middle,
       MemOrder c_mem_order = MemOrder::row_major,
       MemOrder a_mem_order = MemOrder::row_major,
       MemOrder b_mem_order = MemOrder::col_major);
  ~GEMM();
  void Run(cu::DeviceMemory &d_a, cu::DeviceMemory &d_b, cu::DeviceMemory &d_c);
  static dim3 GetDimensions(Precision precision,
                            Variant variant = Variant::opt);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace ccglib::mma
#endif // MMA_GEMM_H_
