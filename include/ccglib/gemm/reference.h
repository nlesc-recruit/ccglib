#ifndef REFERENCE_GEMM_H_
#define REFERENCE_GEMM_H_

#include <ccglib/gemm/mem_order.h>
#include <cuda_fp16.h>

namespace ccglib::reference {
class GEMM {
public:
  virtual void
  Run(const half *a, const half *b, float *c, size_t M, size_t N, size_t K,
      ccglib::mma::MemOrder output_mem_order = ccglib::mma::row_major);
  virtual void
  Run(const float *a, const float *b, float *c, size_t M, size_t N, size_t K,
      ccglib::mma::MemOrder output_mem_order = ccglib::mma::row_major);
  virtual void
  Run(const unsigned *a, const unsigned *b, int *c, size_t M, size_t N,
      size_t K,
      ccglib::mma::MemOrder output_mem_order = ccglib::mma::row_major);
};

} // namespace ccglib::reference

#endif // REFERENCE_GEMM_H_