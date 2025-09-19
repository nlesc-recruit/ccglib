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

bool isRDNA4(cu::Device &device) {
  const std::string arch(device.getArch());
  return (arch.find("gfx12") != std::string::npos);
}

bool isCDNA(cu::Device &device) {
  return (isCDNA1(device) || isCDNA2(device) || isCDNA3(device));
}

bool isVolta(cu::Device &device) {
  const std::string arch(device.getArch());
  return (arch.find("sm_70") != std::string::npos);
}

bool isAda(const std::string &arch) {
  return (arch.find("sm_89") != std::string::npos);
}

bool isHopper(const std::string &arch) {
  return (arch.find("sm_90") != std::string::npos);
}

bool isBlackwell(const std::string &arch) {
  static const std::vector<std::string> blackwell_archs = {"sm_100", "sm_101",
                                                           "sm_120"};
  for (const auto &bw_arch : blackwell_archs) {
    if (arch.find(bw_arch) != std::string::npos) {
      return true;
    }
  }
  return false;
}

bool hasFP8(cu::Device &device) {
  const std::string arch(device.getArch());
  return isBlackwell(arch) || isHopper(arch) || isAda(arch);
}

bool hasFP4(cu::Device &device) {
  const std::string arch(device.getArch());
  return isBlackwell(arch) || isHopper(arch) || isAda(arch);
}

} // namespace ccglib

#endif // ARCH_H_