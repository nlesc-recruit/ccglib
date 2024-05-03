#ifndef VERIFY_H_
#define VERIFY_H_

#include "fpequals.h"

template <typename Tin, typename Tout, unsigned M, unsigned N, unsigned K>
void verify(const Tin *a, const Tin *b, const Tout *c) {
  const std::array<size_t, 3> a_shape = {2, M, K};
  const std::array<size_t, 3> b_shape = {2, N, K};
  const std::array<size_t, 3> c_shape = {2, M, N};

  const size_t a_size = 2 * M * K;
  const size_t b_size = 2 * N * K;
  const size_t c_size = 2 * M * N;

  auto a_view = xt::adapt(a, a_size, xt::no_ownership(), a_shape);
  auto b_view = xt::adapt(b, b_size, xt::no_ownership(), b_shape);
  auto c_view = xt::adapt(c, c_size, xt::no_ownership(), c_shape);

  xt::xtensor<Tout, 3> c_ref = xt::zeros_like(c_view);

  ccglib::reference::GEMM gemm;
  gemm.run(a, b, c_ref.data(), M, N, K);

  for (unsigned m = 0; m < M; m++) {
    for (unsigned n = 0; n < N; n++) {
      std::complex<Tout> ref(c_ref(0, m, n), c_ref(1, m, n));
      std::complex<Tout> tst(c_view(0, m, n), c_view(1, m, n));
      ccglib::test::fpEquals(ref, tst, ccglib::test::getEpsilon<Tin>());
    }
  }
}
#endif // VERIFY_H_