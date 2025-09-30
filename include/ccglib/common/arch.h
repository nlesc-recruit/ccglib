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
  // AMD: only architectures with matrix cores are supported: CDNA or newer,
  // RDNA3 or newer
#if defined(__HIP_PLATFORM_AMD__)
  const std::vector<std::string> prefixes = {"gfx6", "gfx7", "gfx8", "gfx10"};
  if (std::any_of(prefixes.begin(), prefixes.end(), [&](const std::string &p) {
        return arch.find(p) != std::string::npos;
      })) {
    return true;
  }

  // gfx9 is supported if it is CDNA
  return ((arch.find("gfx9") != std::string::npos) && !isCDNA(device));
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