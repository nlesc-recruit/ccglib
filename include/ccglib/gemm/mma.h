#ifndef MMA_GEMM_H_
#define MMA_GEMM_H_

#include <complex>
#include <memory>

#include <ccglib/common/complex_order.h>
#include <ccglib/common/precision.h>
#include <ccglib/gemm/mem_order.h>
#include <ccglib/gemm/variant.h>

#ifdef __HIP__
#include <hip/hip_runtime.h>
#else
#include <cuda.h>
#endif

// Forward declaration of cudawrappers types
namespace cu {
class Device;
class DeviceMemory;
class Stream;
} // namespace cu

namespace ccglib::mma {

class GEMM {
public:
  GEMM(size_t B_, size_t M_, size_t N_, size_t K_, cu::Device &device,
       cu::Stream &stream, Precision precision, Variant Variant = Variant::opt,
       ComplexAxisLocation c_complex_axis_location =
           ComplexAxisLocation::complex_planar,
       MemOrder c_mem_order = MemOrder::row_major,
       MemOrder a_mem_order = MemOrder::row_major,
       MemOrder b_mem_order = MemOrder::col_major,
       std::complex<float> alpha = {1, 0}, std::complex<float> beta = {0, 0});
  ~GEMM();
  void Run(cu::DeviceMemory &d_a, cu::DeviceMemory &d_b, cu::DeviceMemory &d_c);
  static dim3 GetDimensions(Precision precision,
                            Variant variant = Variant::opt);
#ifdef __HIP__
  GEMM(size_t B_, size_t M_, size_t N_, size_t K_, hipDevice_t &device,
       hipStream_t &stream, Precision precision, Variant Variant = Variant::opt,
       ComplexAxisLocation c_complex_axis_location =
           ComplexAxisLocation::complex_planar,
       MemOrder c_mem_order = MemOrder::row_major,
       MemOrder a_mem_order = MemOrder::row_major,
       MemOrder b_mem_order = MemOrder::col_major,
       std::complex<float> alpha = {1, 0}, std::complex<float> beta = {0, 0});
  void Run(hipDeviceptr_t d_a, hipDeviceptr_t d_b, hipDeviceptr_t d_c);
#else
  GEMM(size_t B_, size_t M_, size_t N_, size_t K_, CUdevice &device,
       CUstream &stream, Precision precision, Variant Variant = Variant::opt,
       ComplexAxisLocation c_complex_axis_location =
           ComplexAxisLocation::complex_planar,
       MemOrder c_mem_order = MemOrder::row_major,
       MemOrder a_mem_order = MemOrder::row_major,
       MemOrder b_mem_order = MemOrder::col_major,
       std::complex<float> alpha = {1, 0}, std::complex<float> beta = {0, 0});
  void Run(CUdeviceptr d_a, CUdeviceptr d_b, CUdeviceptr d_c);
#endif

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
  std::unique_ptr<cu::Device> device_;
  std::unique_ptr<cu::Stream> stream_;
};

} // namespace ccglib::mma
#endif // MMA_GEMM_H_
