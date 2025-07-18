#ifndef VERIFY_H_
#define VERIFY_H_

#include <limits.h>
#include <xtensor/xadapt.hpp>
#include <xtensor/xtensor.hpp>

#include <ccglib/gemm/mem_order.h>
#include <ccglib/gemm/reference.h>

#include "fpequals.h"
#include <ccglib/common/precision.h>

template <typename Tin, typename Tout, ccglib::ValueType InputPrecision>
void verify(const Tin *a, const Tin *b, const Tout *c, size_t B, size_t M,
            size_t N, size_t K, ccglib::mma::MemOrder output_mem_order) {
  const size_t kPackingFactor =
      sizeof(Tin) * CHAR_BIT /
      ccglib::ValuePrecision{InputPrecision}.GetBitWidth();

  const std::array<size_t, 4> a_shape = {B, 2, M, K / kPackingFactor};
  const std::array<size_t, 4> b_shape = {B, 2, N, K / kPackingFactor};
  std::array<size_t, 4> c_shape;
  if (output_mem_order == ccglib::mma::row_major) {
    c_shape = {B, 2, M, N};
  } else {
    c_shape = {B, 2, N, M};
  }

  const size_t a_size = B * 2 * M * K / kPackingFactor;
  const size_t b_size = B * 2 * N * K / kPackingFactor;
  const size_t c_size = B * 2 * M * N;

  auto a_view = xt::adapt(a, a_size, xt::no_ownership(), a_shape);
  auto b_view = xt::adapt(b, b_size, xt::no_ownership(), b_shape);
  auto c_view = xt::adapt(c, c_size, xt::no_ownership(), c_shape);

  xt::xtensor<Tout, 4> c_ref = xt::zeros_like(c_view);

  ccglib::reference::GEMM gemm;
  for (size_t batch = 0; batch < B; ++batch) {
    size_t aoffset = batch * 2 * M * K / kPackingFactor;
    size_t boffset = batch * 2 * N * K / kPackingFactor;
    size_t coffset = batch * 2 * M * N;
    gemm.Run(a + aoffset, b + boffset, c_ref.data() + coffset, M, N, K,
             output_mem_order);
  }

  for (size_t b = 0; b < B; b++) {
    for (size_t m = 0; m < M; m++) {
      for (size_t n = 0; n < N; n++) {
        std::complex<Tout> ref;
        std::complex<Tout> tst;
        if (output_mem_order == ccglib::mma::row_major) {
          ref = {c_ref(b, 0, m, n), c_ref(b, 1, m, n)};
          tst = {c_view(b, 0, m, n), c_view(b, 1, m, n)};
        } else {
          ref = {c_ref(b, 0, n, m), c_ref(b, 1, n, m)};
          tst = {c_view(b, 0, n, m), c_view(b, 1, n, m)};
        }
        ccglib::test::fpEquals(ref, tst);
      }
    }
  }
}

#endif // VERIFY_H_