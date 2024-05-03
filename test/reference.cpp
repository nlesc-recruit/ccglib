#include <catch2/catch_test_macros.hpp>
#include <cuda_fp16.h>
#include <xtensor/xadapt.hpp>
#include <xtensor/xcomplex.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

#include "reference/GEMM.h"

namespace ccglib::test {

TEST_CASE("Reference complex") {
  const unsigned M = 3;
  const unsigned N = 3;
  const unsigned K = 2;
  const unsigned COMPLEX = 2;
  // Matrix a row-major a(2,M,K)
  half a[COMPLEX * M * K] = {1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1};
  // Matrix a column-major a(2,M,K)
  half b[COMPLEX * N * K] = {1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1};
  // Matrix c=a*b row-major c(2,M,N)
  std::array<float, COMPLEX * M * N> c_ref = {
      -56, -28, 0, -28, 0, 28, 0, 28, 56, 32, 48, 64, 48, 48, 48, 64, 48, 32};

  std::array<float, COMPLEX * M * N> c_test;

  ccglib::reference::GEMM gemm;
  gemm.run(a, b, &c_test[0], M, N, K);

  REQUIRE(c_ref == c_test);
}

} // namespace ccglib::test
