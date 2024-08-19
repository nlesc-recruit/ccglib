#ifndef TRANSPOSE_H_
#define TRANSPOSE_H_

#include <cudawrappers/cu.hpp>

namespace ccglib::transpose {
class Transpose {
public:
  Transpose(size_t B, size_t M, size_t N, size_t M_chunk, size_t N_chunk,
            size_t nr_bits, cu::Device &device, cu::Stream &stream);
  void Run(cu::HostMemory &h_input, cu::DeviceMemory &d_output);
  void Run(cu::DeviceMemory &d_input, cu::DeviceMemory &d_output);

private:
  size_t B;
  size_t M;
  size_t N;
  size_t M_chunk;
  size_t N_chunk;

  size_t nr_bits;

  cu::Device &device_;
  cu::Stream &stream_;

  void compile_kernel();

  std::shared_ptr<cu::Module> module;
  std::shared_ptr<cu::Function> function;
};

} // namespace ccglib::transpose

#endif // TRANSPOSE_H_