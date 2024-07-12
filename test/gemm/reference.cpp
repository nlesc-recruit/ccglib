#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cuda_fp16.h>
#include <limits.h>
#include <xtensor/xadapt.hpp>
#include <xtensor/xcomplex.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

#include <ccglib/gemm/reference.h>
#include <ccglib/helper.h>

namespace ccglib::test {

template <typename InputType> class TestImpl {
public:
  void runTestReference() {
    const size_t M = 3;
    const size_t N = 3;
    const size_t K = 2;
    const size_t COMPLEX = 2;
    // Matrix a row-major a(2,M,K)
    const InputType a[COMPLEX * M * K] = {1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1};
    // Matrix a column-major a(2,M,K)
    const InputType b[COMPLEX * N * K] = {1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1};
    // Matrix c=a*b row-major c(2,M,N)
    const std::array<float, COMPLEX * M * N> c_ref = {
        -56, -28, 0, -28, 0, 28, 0, 28, 56, 32, 48, 64, 48, 48, 48, 64, 48, 32};

    std::array<float, COMPLEX * M * N> c_test;

    ccglib::reference::GEMM gemm;
    gemm.Run(a, b, &c_test[0], M, N, K);

    REQUIRE(c_ref == c_test);
  }
};

using TestTypes = std::tuple<half, float>;
TEMPLATE_LIST_TEST_CASE_METHOD(TestImpl, "Reference complex float",
                               "[correctness]", TestTypes) {
  SECTION("Simple reference output") { TestImpl<TestType>::runTestReference(); }
}

TEST_CASE("Reference complex binary") {
  const size_t M = 2;
  const size_t N = 2;
  const size_t K = 32;
  const size_t COMPLEX = 2;
  // Inputs are 1-bit values packed into unsigned ints in little-endian order
  const size_t bits_per_sample = sizeof(unsigned) * CHAR_BIT;
  const size_t K_PACKED = ccglib::helper::ceildiv(K, bits_per_sample);
  // A must be row-major, B col-major. Here, C is row-major
  const unsigned a[COMPLEX * M * K_PACKED] = {4007499276, 2587246816,
                                              2368114480, 2180517764};
  const unsigned b[COMPLEX * N * K_PACKED] = {3172811225, 4156143586, 144478092,
                                              808068269};
  const std::array<int, COMPLEX * M * N> c_ref = {-8, 8,   -10, -6,
                                                  8,  -12, -6,  6};

  std::array<int, COMPLEX * M * N> c_test;

  ccglib::reference::GEMM gemm;
  gemm.Run(a, b, &c_test[0], M, N, K);

  REQUIRE(c_ref == c_test);
}

} // namespace ccglib::test
