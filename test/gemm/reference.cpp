#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <limits.h>
#include <xtensor/xadapt.hpp>
#include <xtensor/xcomplex.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

#include <ccglib/fp16.h>
#include <ccglib/gemm/reference.h>
#include <ccglib/helper.h>

namespace ccglib::test {

template <typename InputType> class TestImpl {
public:
  void runTestReference() {
    const size_t M = 3;
    const size_t N = 2;
    const size_t K = 2;
    const size_t COMPLEX = 2;
    // Matrix a row-major a(2,M,K)
    const InputType a[COMPLEX * M * K] = {1., 2., 3., 4., 5., 6.,
                                          6., 5., 4., 3., 2., 1.};
    // Matrix a column-major a(2,M,K)
    const InputType b[COMPLEX * N * K] = {1., 2., 3., 4., 4., 3., 2., 1.};
    // Matrix c=a*b row-major c(2,M,N)
    const std::array<float, COMPLEX * M * N> c_ref_row_major = {
        -34, -6, -14, 14, 6, 34, 26, 42, 34, 34, 42, 26};
    // Matrix c col-major c(2,N,M)
    const std::array<float, COMPLEX * M * N> c_ref_col_major = {
        -34, -14, 6, -6, 14, 34, 26, 34, 42, 42, 34, 26};

    std::array<float, COMPLEX * M * N> c_test;

    ccglib::reference::GEMM gemm;
    gemm.Run(a, b, &c_test[0], M, N, K, ccglib::mma::row_major);
    REQUIRE(c_ref_row_major == c_test);

    gemm.Run(a, b, &c_test[0], M, N, K, ccglib::mma::col_major);
    REQUIRE(c_ref_col_major == c_test);
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
  // A must be row-major, B col-major. C is row-major or col-major
  const auto a = new unsigned[COMPLEX * M * K_PACKED]{4007499276, 2587246816,
                                                      2368114480, 2180517764};
  const auto b = new unsigned[COMPLEX * N * K_PACKED]{3172811225, 4156143586,
                                                      144478092, 808068269};
  const std::array<int, COMPLEX * M * N> c_ref_row_major = {-8, 8,   -10, -6,
                                                            8,  -12, -6,  6};
  const std::array<int, COMPLEX * M * N> c_ref_col_major = {-8, -10, 8,   -6,
                                                            8,  -6,  -12, 6};

  std::array<int, COMPLEX * M * N> c_test;

  ccglib::reference::GEMM gemm;
  gemm.Run(a, b, &c_test[0], M, N, K, ccglib::mma::row_major);
  REQUIRE(c_ref_row_major == c_test);

  gemm.Run(a, b, &c_test[0], M, N, K, ccglib::mma::col_major);
  REQUIRE(c_ref_col_major == c_test);
}

} // namespace ccglib::test
