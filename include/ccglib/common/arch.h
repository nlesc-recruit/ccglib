#ifndef ARCH_H_
#define ARCH_H_

#include <cudawrappers/cu.hpp>

namespace ccglib {
bool isCDNA1(cu::Device &device) {
  const std::string arch(device.getArch());
  return (arch.find("gfx908") != std::string::npos);
}

bool isCDNA2(cu::Device &device) {
  const std::string arch(device.getArch());
  return (arch.find("gfx90a") != std::string::npos);
}

bool isCDNA3(cu::Device &device) {
  const std::string arch(device.getArch());
  return ((arch.find("gfx940") != std::string::npos) ||
          (arch.find("gfx941") != std::string::npos) ||
          (arch.find("gfx942") != std::string::npos));
}

bool isCDNA(cu::Device &device) {
  return (isCDNA1(device) || isCDNA2(device) || isCDNA3(device));
}

bool isVolta(cu::Device &device) {
  const std::string arch(device.getArch());
  return (arch.find("sm_70") != std::string::npos);
}

} // namespace ccglib

#endif // ARCH_H_