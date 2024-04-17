#include <array>

#include <cudawrappers/cu.hpp>

namespace ccglib::mma {
class GEMM {
public:
  enum Variant { basic, opt };

  GEMM(size_t M_, size_t K_, size_t N_, size_t nr_input_bits_,
       size_t nr_output_bits, cu::Device &device_, cu::Stream &stream_,
       Variant Variant = Variant::opt);
  void run(cu::DeviceMemory &d_a, cu::DeviceMemory &d_b, cu::DeviceMemory &d_c);

  // public kernel settings
  static const size_t kMPerBlock = 128;
  static const size_t kNPerBlock = 64;
  static const size_t kKPerWMMA = 16;

private:
  Variant variant_;

  size_t K_;
  size_t M_;
  size_t N_;

  size_t nr_input_bits_;
  const size_t kNrOutputBits = sizeof(float) * 8;

  dim3 threads_;
  dim3 grid_;

  // kernel settings
  const size_t kMPerWarp = 32;
  const size_t kMPerWMMA = 16;

  const size_t kNPerWarp = 32;
  const size_t kNPerWMMA = 16;

  const size_t kWarpSize = 32;
  const size_t kNBuffer = 4;

  cu::Device &device_;
  cu::Stream &stream_;

  void compile_kernel();

  std::unique_ptr<cu::Module> module_;
  std::unique_ptr<cu::Function> function_;
};

} // namespace ccglib::mma
