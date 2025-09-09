#ifndef MMA_PRECISION_H_
#define MMA_PRECISION_H_

#include <ccglib/fp16.h>
#include <ccglib/fp8.h>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>

namespace ccglib {

enum ValueType { int1, int32, float8e4m3, float16, float32 };

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
    case ValueType::float8e4m3:
      return CHAR_BIT;
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

#endif // MMA_PRECISION_H_