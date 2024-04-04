#include <array>

#include <cudawrappers/cu.hpp>

namespace ccglib::mma {
class GEMM {
public:
  enum Variant { basic, opt };

  GEMM(size_t beams_, size_t samples_, size_t frames_, size_t nr_input_bits_,
       size_t nr_output_bits, cu::Device &device_, cu::Stream &stream_,
       Variant Variant = Variant::opt);
  void run(cu::DeviceMemory &d_a, cu::DeviceMemory &d_b, cu::DeviceMemory &d_c);

  // public kernel settings
  static const size_t kBeamsPerBlock = 128;
  static const size_t kFramesPerBlock = 64;
  static const size_t kSamplesPerWMMA = 16;

private:
  Variant variant_;

  size_t samples_;
  size_t beams_;
  size_t frames_;

  size_t nr_input_bits_;
  const size_t kNrOutputBits = sizeof(float) * 8;

  dim3 threads_;
  dim3 grid_;

  // kernel settings
  const size_t kBeamsPerWarp = 32;
  const size_t kBeamsPerWMMA = 16;

  const size_t kFramesPerWarp = 32;
  const size_t kFramesPerWMMA = 16;

  const size_t kWarpSize = 32;
  const size_t kNBuffer = 4;

  cu::Device &device_;
  cu::Stream &stream_;

  void compile_kernel();

  std::unique_ptr<cu::Module> module_;
  std::unique_ptr<cu::Function> function_;
};

} // namespace ccglib::mma
