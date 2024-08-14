#ifndef PACKING_H_
#define PACKING_H_

#include <cudawrappers/cu.hpp>

namespace ccglib::packing {
enum Direction { pack, unpack };
enum ComplexAxisLocation { complex_first, complex_last };

class Packing {
public:
  Packing(size_t N, cu::Device &device, cu::Stream &stream);
  void Run(cu::HostMemory &h_input, cu::DeviceMemory &d_output,
           Direction direction,
           ComplexAxisLocation complex_axis_location = complex_first);
  void Run(cu::DeviceMemory &d_input, cu::DeviceMemory &d_output,
           Direction direction,
           ComplexAxisLocation complex_axis_location = complex_first);

private:
  size_t N_;
  cu::Device &device_;
  cu::Stream &stream_;

  void compile_kernel();

  std::shared_ptr<cu::Module> module_;
  std::shared_ptr<cu::Function> function_pack_;
  std::shared_ptr<cu::Function> function_unpack_;
};

} // namespace ccglib::packing

#endif // PACKING_H_
