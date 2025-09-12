#ifndef PACKING_H_
#define PACKING_H_

#include <memory>

#ifdef __HIP__
#include <hip/hip_runtime.h>
#else
#include <cuda.h>
#endif

#include <ccglib/common/complex_order.h>
#include <ccglib/common/direction.h>

// Forward declaration of cudawrappers types
namespace cu {
class Device;
class DeviceMemory;
class HostMemory;
class Stream;
} // namespace cu

namespace ccglib::packing {

class Packing {
public:
  Packing(size_t N, Direction direction, cu::Device &device, cu::Stream &stream,
          ComplexAxisLocation input_complex_axis_location =
              ComplexAxisLocation::complex_planar);
  ~Packing();
  void Run(cu::HostMemory &h_input, cu::DeviceMemory &d_output);
  void Run(cu::DeviceMemory &d_input, cu::DeviceMemory &d_output);
#ifdef __HIP__
  Packing(size_t N, Direction direction, hipDevice_t &device,
          hipStream_t &stream,
          ComplexAxisLocation input_complex_axis_location =
              ComplexAxisLocation::complex_planar);
  void Run(unsigned char *h_input, hipDeviceptr_t d_output);
  void Run(hipDeviceptr_t d_input, hipDeviceptr_t d_output);
#else
  Packing(size_t N, Direction direction, CUdevice &device, CUstream &stream,
          ComplexAxisLocation input_complex_axis_location =
              ComplexAxisLocation::complex_planar);
  void Run(unsigned char *h_input, CUdeviceptr d_output);
  void Run(CUdeviceptr d_input, CUdeviceptr d_output);
#endif

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
  std::unique_ptr<cu::Device> device_;
  std::unique_ptr<cu::Stream> stream_;
  size_t N_;
};

} // namespace ccglib::packing

#endif // PACKING_H_
