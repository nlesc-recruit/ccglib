#ifndef MMA_KERNEL_H_
#define MMA_KERNEL_H_

#include <cuda_runtime.h>

#include <string>

#include <ccglib/gemm/precision.h>
#include <ccglib/gemm/variant.h>

namespace ccglib::mma {

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

} // namespace ccglib::mma

#endif // MMA_KERNEL_H_