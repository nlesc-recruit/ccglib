#include <array>

#include <cudawrappers/cu.hpp>

namespace ccglib::mma {
class GEMM {
public:
  GEMM(size_t beams, size_t samples, size_t frames, size_t nr_input_bits,
       size_t nr_output_bits, cu::Device &device, cu::Stream &stream);
  void run(cu::DeviceMemory &d_a, cu::DeviceMemory &d_b, cu::DeviceMemory &d_c);

  // public kernel settings
  static const size_t kBeamsPerBlock = 128;
  static const size_t kFramesPerBlock = 64;
  static const size_t kSamplesPerWMMA = 16;

private:
  size_t samples;
  size_t beams;
  size_t frames;

  size_t nr_input_bits;
  size_t nr_output_bits;

  dim3 threads;
  dim3 grid;

  // kernel settings
  const size_t beams_per_warp = 32;
  const size_t beams_per_wmma = 16;

  const size_t frames_per_warp = 32;
  const size_t frames_per_wmma = 16;

  const size_t warp_size = 32;
  const size_t nbuffer = 4;

  cu::Device &device;
  cu::Stream &stream;

  void compile_kernel();

  std::unique_ptr<cu::Module> module;
  std::unique_ptr<cu::Function> function;
};

} // namespace ccglib::mma
