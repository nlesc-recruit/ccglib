#ifndef TRANSPOSE_H_
#define TRANSPOSE_H_

#include <ccglib/transpose/complex_order.h>

// Forward declaration of cudawrappers types
namespace cu {
class Device;
class DeviceMemory;
class HostMemory;
class Stream;
} // namespace cu
namespace ccglib::transpose {
class Transpose {
public:
  Transpose(size_t B, size_t M, size_t N, size_t M_chunk, size_t N_chunk,
            size_t nr_bits, cu::Device &device, cu::Stream &stream,
            ComplexAxisLocation input_complex_axis_location =
                ComplexAxisLocation::complex_middle);
  ~Transpose();
  void Run(cu::HostMemory &h_input, cu::DeviceMemory &d_output);
  void Run(cu::DeviceMemory &d_input, cu::DeviceMemory &d_output);

#ifdef __HIP__
  Transpose(size_t B, size_t M, size_t N, size_t M_chunk, size_t N_chunk,
            size_t nr_bits, hipDevice_t &device, hipStream_t &stream,
            ComplexAxisLocation input_complex_axis_location =
                ComplexAxisLocation::complex_middle);
  void Run(const void *h_input, hipDeviceptr_t d_output);
  void Run(hipDeviceptr_t d_input, hipDeviceptr_t d_output);
#else
  Transpose(size_t B, size_t M, size_t N, size_t M_chunk, size_t N_chunk,
            size_t nr_bits, CUdevice &device, CUstream &stream,
            ComplexAxisLocation input_complex_axis_location =
                ComplexAxisLocation::complex_middle);
  void Run(const void *h_input, CUdeviceptr d_output);
  void Run(CUdeviceptr d_input, CUdeviceptr d_output);
#endif

protected:
  size_t B_;
  size_t M_;
  size_t N_;
  size_t M_chunk_;
  size_t N_chunk_;

  size_t nr_bits_;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
  std::unique_ptr<cu::Device> device_;
  std::unique_ptr<cu::Stream> stream_;
};

} // namespace ccglib::transpose

#endif // TRANSPOSE_H_