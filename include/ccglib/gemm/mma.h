#ifndef MMA_GEMM_H_
#define MMA_GEMM_H_

#include <array>

#include <cudawrappers/cu.hpp>

namespace ccglib::mma {
enum Precision { float16, int1 };
enum Variant { basic, opt };

class Kernel {
public:
  struct Parameters {
    size_t m_per_block;
    size_t m_per_warp;
    size_t m_per_wmma;

    size_t n_per_block;
    size_t n_per_warp;
    size_t n_per_wmma;

    size_t k_per_wmma;

    size_t warp_size;
    size_t nbuffer;
  };

  Kernel(Precision precision, Variant variant);

  dim3 GetThreads() const;
  std::string GetSource() const;
  Parameters GetParameters() const { return parameters_; };
  std::string GetName() const;

private:
  Precision precision_;
  Variant variant_;
  Parameters parameters_;

  template <Precision P> Parameters GetParameters() const;

  void SetParameters(Precision precision);

  template <Precision P> std::string GetSource() const;
};

class GEMM {
public:
  GEMM(size_t B_, size_t M_, size_t K_, size_t N_, size_t nr_input_bits_,
       cu::Device &device_, cu::Stream &stream_, Precision precision,
       Variant Variant = Variant::opt);
  void run(cu::DeviceMemory &d_a, cu::DeviceMemory &d_b, cu::DeviceMemory &d_c);

  // public kernel settings
  size_t kMPerBlock;
  size_t kNPerBlock;
  size_t kKPerWMMA;

private:
  Variant variant_;
  const Kernel &kernel_;

  size_t B_;
  size_t K_;
  size_t M_;
  size_t N_;

  size_t nr_input_bits_;

  dim3 threads_;
  dim3 grid_;

  cu::Device &device_;
  cu::Stream &stream_;

  void compile_kernel();

  std::unique_ptr<cu::Module> module_;
  std::unique_ptr<cu::Function> function_;
};

} // namespace ccglib::mma

#endif // MMA_GEMM_H_