#ifndef MMA_KERNEL_H_
#define MMA_KERNEL_H_

#if defined(__HIP__)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

#include <string>

#include <ccglib/gemm/variant.h>
#include <ccglib/precision.h>
#include <cudawrappers/cu.hpp>

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

    size_t nbuffer;
  };

  Kernel(Precision precision, const Variant &variant);

  dim3 GetThreads(cu::Device &device) const;
  std::string GetSource() const;
  Parameters GetParameters() const { return parameters_; };
  std::string GetName() const;
  const Precision &GetPrecision() const { return precision_; }

private:
  const Precision precision_;
  const Variant &variant_;
  Parameters parameters_;

  template <ValueType T> Parameters GetCompileParameters() const;

  void SetParameters(const Precision precision);

  template <ValueType T> std::string GetSource() const;
};

} // namespace ccglib::mma

#endif // MMA_KERNEL_H_