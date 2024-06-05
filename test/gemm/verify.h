#ifndef VERIFY_H_
#define VERIFY_H_

#include <xtensor/xadapt.hpp>
#include <xtensor/xtensor.hpp>

#include "fpequals.h"

template <typename Tin, typename Tout>
void verify(const Tin *a, const Tin *b, const Tout *c, unsigned B, unsigned M,
            unsigned N, unsigned K) {
  const std::array<size_t, 4> a_shape = {B, 2, M, K};
  const std::array<size_t, 4> b_shape = {B, 2, N, K};
  const std::array<size_t, 4> c_shape = {B, 2, M, N};

  const size_t a_size = B * 2 * M * K;
  const size_t b_size = B * 2 * N * K;
  const size_t c_size = B * 2 * M * N;

  auto a_view = xt::adapt(a, a_size, xt::no_ownership(), a_shape);
  auto b_view = xt::adapt(b, b_size, xt::no_ownership(), b_shape);
  auto c_view = xt::adapt(c, c_size, xt::no_ownership(), c_shape);

  xt::xtensor<Tout, 4> c_ref = xt::zeros_like(c_view);

  ccglib::reference::GEMM gemm;
  for (size_t batch = 0; batch < B; ++batch) {
    size_t aoffset = batch * 2 * M * K;
    size_t boffset = batch * 2 * N * K;
    size_t coffset = batch * 2 * M * N;
    gemm.Run(a + aoffset, b + boffset, c_ref.data() + coffset, M, N, K);
  }

  for (unsigned b = 0; b < B; b++) {
    for (unsigned m = 0; m < M; m++) {
      for (unsigned n = 0; n < N; n++) {
        std::complex<Tout> ref(c_ref(b, 0, m, n), c_ref(b, 1, m, n));
        std::complex<Tout> tst(c_view(b, 0, m, n), c_view(b, 1, m, n));
        ccglib::test::fpEquals(ref, tst, ccglib::test::getEpsilon<Tin>());
      }
    }
  }
}

#endif // VERIFY_H_