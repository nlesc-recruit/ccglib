#ifndef VALUE_TYPE_H_
#define VALUE_TYPE_H_

// Ensure headers are inlined by including them locally in device compilation
// pass
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#include "ccglib/fp16.h"
#else
#include <ccglib/fp16.h>
#endif

namespace ccglib {

enum ValueType { int1, int32, bfloat16, float16, float32 };

constexpr size_t __host__ __device__ CalculateBitWidth(ValueType type) {
  switch (type) {
  case ValueType::int1:
    return 1;
  case ValueType::bfloat16:
  case ValueType::float16:
    return 16;
  case ValueType::float32:
  case ValueType::int32:
    return 32;
  default:
    // Default case, shouldn't happen
    return 0;
  }
}

} // namespace ccglib

#endif // VALUE_TYPE_H_
