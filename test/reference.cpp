#include <catch2/catch_test_macros.hpp>
#include <cuda_fp16.h>
#include <xtensor/xadapt.hpp>
#include <xtensor/xcomplex.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

#include "reference/GEMM.h"

TEST_CASE("Reference complex") {
  // M=3, N=3, K=2
  // Matrix a row-major a(2,M,K)
  half a[] = {1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1};
  // Matrix a column-major a(2,M,K)
  half b[] = {1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1};
  // Matrix c=a*b row-major c(2,M,N)
  std::array<float, 18> c_ref = {-56, -28, 0,  -28, 0,  28, 0,  28, 56,
                                 32,  48,  64, 48,  48, 48, 64, 48, 32};

  std::array<float, 18> c_test;

  ccglib::reference::GEMM gemm;
  gemm.run(a, b, &c_test[0], 3, 3, 2);

  REQUIRE(c_ref == c_test);
}
