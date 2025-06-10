#ifndef VALUE_PRECISION_H_
#define VALUE_PRECISION_H_

#include <ccglib/fp16.h>

#include "value_type.h"

namespace ccglib {

struct ValuePrecision {
  constexpr ValuePrecision(ValueType type)
      : type(type), num_bits(CalculateBitWidth()) {}

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
  constexpr size_t CalculateBitWidth() const {
    switch (type) {
    case ValueType::int1:
      return 1;
    case ValueType::int32:
      return CHAR_BIT * sizeof(unsigned);
    case ValueType::float16:
      return CHAR_BIT * sizeof(half);
    case ValueType::float32:
      return CHAR_BIT * sizeof(float);
    }
    // Default case, shouldn't happen
    throw std::invalid_argument("Invalid data type");
  }

  const ValueType type;
  const size_t num_bits;
};

} // namespace ccglib

#endif // VALUE_PRECISION_H_