#ifndef REFERENCE_GEMM_H_
#define REFERENCE_GEMM_H_

#include <cuda_fp16.h>

namespace ccglib::reference {
class GEMM {
public:
  virtual void run(const half *a, const half *b, float *c, unsigned M,
                   unsigned N, unsigned K);
  virtual void run(const unsigned *a, const unsigned *b, int *c, unsigned M,
                   unsigned N, unsigned K);
};

} // namespace ccglib::reference

#endif // REFERENCE_GEMM_H_