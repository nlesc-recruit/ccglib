#ifndef VALUE_TYPE_H_
#define VALUE_TYPE_H_

// Only include header in host compilation pass
#if !(defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__))
#include <ccglib/fp16.h>
#endif

namespace ccglib {

enum ValueType { int1, int32, float16, float32 };

constexpr size_t CalculateBitWidth(ValueType type) {
  switch (type) {
  case ValueType::int1:
    return 1;
  case ValueType::int32:
    return CHAR_BIT * sizeof(unsigned);
  case ValueType::float16:
    return CHAR_BIT * sizeof(half);
  case ValueType::float32:
    return CHAR_BIT * sizeof(float);
  default:
    // Default case, shouldn't happen
    return 0;
  }
}

} // namespace ccglib

#endif // VALUE_TYPE_H_