#ifndef ARCH_H_
#define ARCH_H_

#include <cudawrappers/cu.hpp>

namespace ccglib {

int getComputeVersion(cu::Device &device) {
  const int major =
      device.getAttribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
  const int minor =
      device.getAttribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
  return (10 * major + minor);
}

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
  return (arch.find("gfx11") != std::string::npos);
}

bool isCDNA(cu::Device &device) {
  return (isCDNA1(device) || isCDNA2(device) || isCDNA3(device));
}

bool isVolta(cu::Device &device) {
  const std::string arch(device.getArch());
  return (arch.find("sm_70") != std::string::npos);
}

bool hasFP8(cu::Device &device) {
  // In case of AMD, FP8 is only supported in software on CDNA3 GPUs.
  // RDMA4 offers hardware support for FP8.
  return ((getComputeVersion(device) >= 89) || isRDNA4(device) ||
          isCDNA3(device));
}

bool hasFP4(cu::Device &device) {
  // Only supported in hardware from compute capability 10.0 (Hopper) onwards.
  // Emulated in software on earlier architectures (>= 89).
  return (getComputeVersion(device) >= 89);
}

} // namespace ccglib

#endif // ARCH_H_