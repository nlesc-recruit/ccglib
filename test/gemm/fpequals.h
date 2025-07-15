#ifndef FPEQUALS_H_
#define FPEQUALS_H_

#include <complex>
#include <iostream>
#include <limits>

#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <ccglib/bf16.h>
#include <ccglib/fp16.h>

namespace ccglib::test {

template <typename T> constexpr float getEpsilon() {
  if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, float>) {
    // float16 uses an 11-bit mantissa (of which 10 bits are stored)
    // within the kernel, float32 is converted to tf32
    // fp16 and tf32 use an 10-bit mantissa
    // the precision for normal numbers is therefore 2^-10
    return 0.000976562;
  } else if constexpr (std::is_same_v<T, bf16>) {
    // bfloat16 uses an 8-bit mantissa (of which 7 bits are stored)
    // the precision for normal numbers is therefore 2^-7
    // Note: this epsilon exists already in e.g. HIP, but is not constexpr so
    // cannot be reused here
    return 0.0078125;
  }
  return std::numeric_limits<T>::epsilon();
}

template <typename T> void fpEquals(T x, T y, size_t K = 1) {
  constexpr float epsilon = getEpsilon<T>();

  if constexpr (std::is_same_v<T, bf16> || std::is_same_v<T, half>) {
    // We need to upcast since Catch2 cannot print bfloat16/half types in case
    // of test failure.
    const float x_conv = static_cast<float>(x);
    const float y_conv = static_cast<float>(y);

    // We are more lenient in WithinAbs since we use a less precise type.
    REQUIRE_THAT(y_conv, Catch::Matchers::WithinAbs(x_conv, epsilon * K) ||
                             Catch::Matchers::WithinRel(x_conv, epsilon * 100));
  } else {
    REQUIRE_THAT(y, Catch::Matchers::WithinAbs(x, epsilon * K) ||
                        Catch::Matchers::WithinRel(x, epsilon * 100));
  }
}

template <typename T>
void fpEquals(std::complex<T> x, std::complex<T> y, size_t K = 1) {
  fpEquals(x.real(), y.real(), K);
  fpEquals(x.imag(), y.imag(), K);
}

} // namespace ccglib::test
#endif // FPEQUALS_H_
