#ifndef FPEQUALS_H_
#define FPEQUALS_H_

#include <complex>
#include <iostream>
#include <limits>

#include <catch2/catch_test_macros.hpp>
#include <ccglib/bf16.h>
#include <ccglib/fp16.h>
#include <ccglib/fp8.h>

namespace ccglib::test {

template <typename T> constexpr float getEpsilon() {
  if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
    // fp8 uses a 3-bit mantissa
    // the precision for normal numbers is therefore 2^-3
    return 0.125;
  } else if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, float>) {
    // float16 uses an 11-bit mantissa (of which 10 bits are stored)
    // within the kernel, float32 is converted to tf32
    // fp16 and tf32 use an 10-bit mantissa
    // the precision for normal numbers is therefore 2^-10
    return 0.000976562;
  } else if constexpr (std::is_same_v<T, bf16>) {
    // bfloat16 uses an 8-bit mantissa (of which 7 bits are stored)
    // the precision for normal numbers is therefore 2^-7
    return 0.0078125;
  }
  return static_cast<float>(std::numeric_limits<T>::epsilon());
}

template <typename T, typename T_EPSILON = T> void fpEquals(T x, T y) {
  // Follow the same approach as rocWMMA: the max relative error should be
  // < 10 * epsilon. If the output is a narrow type, the downcast results in a
  // loss of precision and the tolerance is increased to 100 * epsilon.
  constexpr double epsilon = static_cast<double>(getEpsilon<T_EPSILON>());
  constexpr double max_rel_error =
      (sizeof(T) < sizeof(float32) ? 100 : 10) * epsilon;

  std::cout << "x: " << static_cast<double>(x)
            << ", y: " << static_cast<double>(y) << std::endl;
  const double x_conv = static_cast<double>(x);
  const double y_conv = static_cast<double>(y);

  const double numerator = std::fabs(x_conv - y_conv);
  const double divisor = std::fabs(x_conv) + std::fabs(y_conv) + 1.0;
  const double rel_error = numerator / divisor;
  REQUIRE(rel_error <= max_rel_error);
}

template <typename T, typename T_EPSILON = T>
void fpEquals(const std::complex<T> &x, const std::complex<T> &y) {
  fpEquals<T, T_EPSILON>(x.real(), y.real());
  fpEquals<T, T_EPSILON>(x.imag(), y.imag());
}

} // namespace ccglib::test
#endif // FPEQUALS_H_
