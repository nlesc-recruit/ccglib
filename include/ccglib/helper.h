#ifndef HELPER_H_
#define HELPER_H_

#include <cudawrappers/cu.hpp>

namespace ccglib::helper {
inline int get_capability(cu::Device &device) {
  return 10 *
             device
                 .getAttribute<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR>() +
         device.getAttribute<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR>();
}

template <typename T> inline T ceildiv(T a, T b) {
  // Only for positive a and b
  return ((a) / (b) + ((a) % (b) != 0));
}

inline size_t ceildiv(size_t a, unsigned int b) {
  return ceildiv(a, static_cast<size_t>(b));
}

} // namespace ccglib::helper

#endif // HELPER_H_