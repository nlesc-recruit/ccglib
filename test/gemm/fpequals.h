#ifndef FPEQUALS_H_
#define FPEQUALS_H_

#include <iostream>
#include <limits>

#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <ccglib/fp16.h>

namespace ccglib::test {

template <typename T> constexpr float getEpsilon() {
  if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, float>) {
    // float16 uses an 11-bit mantissa (of which 10 bits are stored)
    // within the kernel, float32 is converted to tf32
    // tf32 uses the same mantissa as the half-precision math
    // the precision for normal numbers is therefore 2^-10
    return 0.000976562;
  }
  return std::numeric_limits<T>::epsilon();
}

template <typename T> void fpEquals(T x, T y, size_t K) {
  constexpr float epsilon = getEpsilon<T>();

  if constexpr (std::is_same_v<T, half>) {
    // We need to upcast since Catch2 cannot print float16/half types in case of
    // test failure.
    const float x_conv = __half2float(x);
    const float y_conv = __half2float(y);

    // We are more lenient in WithinAbs since we use a less precise type.
    REQUIRE_THAT(y_conv, Catch::Matchers::WithinAbs(x_conv, epsilon * K) ||
                             Catch::Matchers::WithinRel(x_conv, epsilon * 100));
  } else {
    REQUIRE_THAT(y, Catch::Matchers::WithinAbs(x, epsilon) ||
                        Catch::Matchers::WithinRel(x, epsilon * 100));
  }
}

template <typename T>
void fpEquals(std::complex<T> x, std::complex<T> y, size_t K) {
  fpEquals(x.real(), y.real(), K);
  fpEquals(x.imag(), y.imag(), K);
}

} // namespace ccglib::test
#endif // FPEQUALS_H_
