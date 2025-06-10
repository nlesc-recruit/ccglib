#ifndef PRECISION_H_
#define PRECISION_H_

#include "value_precision.h"
#include "value_type.h"

namespace ccglib {

// Represents a precision for ccglib operations.
// This struct can hold the information about both the data type and the bit
// size. It is used to determine the memory layout of the
// input and output data.
struct Precision {
  constexpr Precision(const ValueType input_type)
      : input_type(input_type), output_type(ValueType::float32) {}

  constexpr Precision(const ValueType input_type, const ValueType output_type)
      : input_type(input_type), output_type(output_type) {}

  constexpr Precision(const ValuePrecision input_type)
      : input_type(input_type), output_type(ValueType::float32) {}

  constexpr Precision(const ValuePrecision input_type,
                      const ValuePrecision output_type)
      : input_type(input_type), output_type(output_type) {}

  constexpr inline size_t GetInputBits() const {
    return input_type.GetBitWidth();
  }

  constexpr inline size_t GetOutputBits() const {
    return output_type.GetBitWidth();
  }

  const ValuePrecision input_type;
  const ValuePrecision output_type;
};
} // namespace ccglib

#endif // PRECISION_H_