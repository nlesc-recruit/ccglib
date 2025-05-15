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

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace ccglib::transpose

#endif // TRANSPOSE_H_