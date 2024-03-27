#include <omp.h>
#include <xtensor/xadapt.hpp>
#include <xtensor/xcomplex.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

#include "GEMM.h"

namespace {
template <typename Tin, typename Tout>
void run(const Tin *a, const Tin *b, Tout *c, unsigned M, unsigned N,
         unsigned K) {
  const std::array<size_t, 3> a_shape = {2, M, K};
  const std::array<size_t, 3> b_shape = {2, N, K};
  const std::array<size_t, 3> c_shape = {2, M, N};

  const size_t a_size = 2 * M * K;
  const size_t b_size = 2 * N * K;
  const size_t c_size = 2 * M * N;

  auto a_view = xt::adapt(a, a_size, xt::no_ownership(), a_shape);
  auto b_view = xt::adapt(b, b_size, xt::no_ownership(), b_shape);
  auto c_view = xt::adapt(c, c_size, xt::no_ownership(), c_shape);

#pragma omp parallel for collapse(2)
  for (unsigned m = 0; m < M; ++m) {
    for (unsigned n = 0; n < N; ++n) {
      Tout sum_real = 0;
      Tout sum_imag = 0;

      for (unsigned k = 0; k < K; ++k) {
        const Tout a_real = a_view(0, m, k);
        const Tout a_imag = a_view(1, m, k);
        const Tout b_real = b_view(0, n, k);
        const Tout b_imag = b_view(1, n, k);

        sum_real += a_real * b_real - a_imag * b_imag;
        sum_imag += a_real * b_imag + a_imag * b_real;
      }

      c_view(0, m, n) = sum_real;
      c_view(1, m, n) = sum_imag;
    }
  }
}
} // namespace

namespace ccglib::reference {
void GEMM::run(const half *a, const half *b, float *c, unsigned M, unsigned N,
               unsigned K) {
  ::run<half, float>(a, b, c, M, N, K);
}

} // end namespace ccglib::reference