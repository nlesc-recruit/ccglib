#ifndef FPEQUALS_H_
#define FPEQUALS_H_

#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <limits>

namespace ccglib::test {

template <typename T> void fpEquals(T x, T y, float epsilon) {
  REQUIRE_THAT(y, Catch::Matchers::WithinAbs(x, epsilon) ||
                      Catch::Matchers::WithinRel(x, epsilon * 100));
}

template <typename T>
void fpEquals(std::complex<T> x, std::complex<T> y, float epsilon) {
  fpEquals(x.real(), y.real(), epsilon);
  fpEquals(x.imag(), y.imag(), epsilon);
}

template <typename T> float getEpsilon() {
  if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, float>) {
    // float16 uses an 11-bit mantissa (of which 10 bits are stored)
    // within the kernel, float32 is converted to tf32
    // tf32 uses the same mantissa as the half-precision math
    // the precision for normal numbers is therefore 2^-10
    return 0.000976562;
  }
  return std::numeric_limits<T>::epsilon();
}

} // namespace ccglib::test
#endif // FPEQUALS_H_
