#ifndef ARCH_H_
#define ARCH_H_

#include <cudawrappers/cu.hpp>

namespace ccglib {

static inline bool isCDNA1(cu::Device &device) {
  const std::string arch(device.getArch());
  return (arch.find("gfx908") != std::string::npos);
}

static inline bool isCDNA2(cu::Device &device) {
  const std::string arch(device.getArch());
  return (arch.find("gfx90a") != std::string::npos);
}

static inline bool isCDNA3(cu::Device &device) {
  const std::string arch(device.getArch());
  return (arch.find("gfx94") != std::string::npos);
}

static inline bool isCDNA4(cu::Device &device) {
  const std::string arch(device.getArch());
  return (arch.find("gfx95") != std::string::npos);
}

static inline bool isRDNA4(cu::Device &device) {
  const std::string arch(device.getArch());
  return (arch.find("gfx12") != std::string::npos);
}

static inline bool isCDNA(cu::Device &device) {
  return (isCDNA1(device) || isCDNA2(device) || isCDNA3(device) ||
          isCDNA4(device));
}

static inline bool isVolta(cu::Device &device) {
  const std::string arch(device.getArch());
  return (arch.find("sm_70") != std::string::npos);
}

static inline bool isAda(const std::string &arch) {
  return (arch.find("sm_89") != std::string::npos);
}

static inline bool isHopper(const std::string &arch) {
  return (arch.find("sm_90") != std::string::npos);
}

static inline bool isBlackwell(const std::string &arch) {
  static const std::vector<std::string> blackwell_archs = {"sm_100", "sm_101",
                                                           "sm_120"};
  for (const auto &bw_arch : blackwell_archs) {
    if (arch.find(bw_arch) != std::string::npos) {
      return true;
    }
  }
  return false;
}

static inline bool hasFP8(cu::Device &device) {
  const std::string arch(device.getArch());
  return (isBlackwell(arch) || isHopper(arch) || isAda(arch)) ||
         isRDNA4(device) || isCDNA3(device);
}

static inline bool hasFP4(cu::Device &device) {
  const std::string arch(device.getArch());
  return (isBlackwell(arch) || isHopper(arch) || isAda(arch));
}

static inline bool isUnsupported(cu::Device &device) {
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