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

template <typename T> inline int ceildiv(T a, T b) {
  // Only for positive a and b
  return ((a) / (b) + ((a) % (b) != 0));
}

} // namespace ccglib::helper

#endif // HELPER_H_