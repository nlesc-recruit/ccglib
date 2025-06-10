#ifndef PACKING_H_
#define PACKING_H_

#include <memory>

#ifdef __HIP__
#include <hip/hip_runtime.h>
#else
#include <cuda.h>
#endif

#include <ccglib/common/complex_order.h>

// Forward declaration of cudawrappers types
namespace cu {
class Device;
class DeviceMemory;
class HostMemory;
class Stream;
} // namespace cu

namespace ccglib::packing {
enum Direction { pack, unpack };

class Packing {
public:
  Packing(size_t N, cu::Device &device, cu::Stream &stream);
  ~Packing();
  void Run(cu::HostMemory &h_input, cu::DeviceMemory &d_output,
           Direction direction,
           ComplexAxisLocation input_complex_axis_location = complex_planar);
  void Run(cu::DeviceMemory &d_input, cu::DeviceMemory &d_output,
           Direction direction,
           ComplexAxisLocation input_complex_axis_location = complex_planar);
#ifdef __HIP__
  Packing(size_t N, hipDevice_t &device, hipStream_t &stream);
  void Run(unsigned char *h_input, hipDeviceptr_t d_output, Direction direction,
           ComplexAxisLocation input_complex_axis_location = complex_planar);
  void Run(hipDeviceptr_t d_input, hipDeviceptr_t d_output, Direction direction,
           ComplexAxisLocation input_complex_axis_location = complex_planar);
#else
  Packing(size_t N, CUdevice &device, CUstream &stream);
  void Run(unsigned char *h_input, CUdeviceptr d_output, Direction direction,
           ComplexAxisLocation input_complex_axis_location = complex_planar);
  void Run(CUdeviceptr d_input, CUdeviceptr d_output, Direction direction,
           ComplexAxisLocation input_complex_axis_location = complex_planar);
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
