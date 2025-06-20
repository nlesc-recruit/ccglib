#ifndef VALUE_PRECISION_H_
#define VALUE_PRECISION_H_

#include <ccglib/fp16.h>

#include "value_type.h"

namespace ccglib {

struct ValuePrecision {
  constexpr ValuePrecision(ValueType type)
      : type(type), num_bits(CalculateBitWidth(type)) {}

  constexpr ValuePrecision(ValueType type, size_t num_bits)
      : type(type), num_bits(num_bits) {}

  constexpr bool operator==(const ValuePrecision &other) const {
    return type == other.type;
  }
  constexpr bool operator==(const ValueType &other) const {
    return type == other;
  }

  // Conversion operator to ValueType
  constexpr operator ValueType() const { return type; }

  // Get the bit-width of the current data type
  constexpr size_t GetBitWidth() const { return num_bits; }

private:
  const ValueType type;
  const size_t num_bits;
};

} // namespace ccglib

#endif // VALUE_PRECISION_H_