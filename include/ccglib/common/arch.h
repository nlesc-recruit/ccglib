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
  return (arch.find("gfx94") != std::string::npos);
}

bool isCDNA4(cu::Device &device) {
  const std::string arch(device.getArch());
  return (arch.find("gfx95") != std::string::npos);
}

bool isCDNA(cu::Device &device) {
  return (isCDNA1(device) || isCDNA2(device) || isCDNA3(device) ||
          isCDNA4(device));
}

bool isVolta(cu::Device &device) {
  const std::string arch(device.getArch());
  return (arch.find("sm_70") != std::string::npos);
}

bool isUnsupported(cu::Device &device) {
  const std::string arch(device.getArch());
  // AMD: unsupported are gfx < 8, Vega gfx9, RDNA2=gfx10
#if defined(__HIP_PLATFORM_AMD__)
  if (arch.find("gfx6") != std::string::npos)
    return true;
  if (arch.find("gfx7") != std::string::npos)
    return true;
  if (arch.find("gfx8") != std::string::npos)
    return true;
  if ((arch.find("gfx9") != std::string::npos) && !isCDNA(device))
    return true;
  if (arch.find("gfx10") != std::string::npos)
    return true;
#else
  // NVIDIA: older than Volta is not supported
  const int arch_numeric = std::stoi(arch.substr(3));
  if (arch_numeric < 70)
    return true;
#endif
  return false;
}
} // namespace ccglib

#endif // ARCH_H_