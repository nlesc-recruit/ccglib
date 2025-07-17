#if defined(__HIP__)
// rocWMMA uses hip_bfloat16, defined in hip_bfloat16.h. HIP uses
// __hip_bfloat16, defined in hip_bf16.h, in e.g. type cast operators. There is
// no type cast between hip_bfloat16 and __hip_bfloat16, so we resort to using
// a different type on the host and device, assuming the underlying data is
// identical.
#ifdef __HIP_DEVICE_COMPILE__
#include <hip/hip_bfloat16.h>
using bf16 = hip_bfloat16;
#else
#include <hip/hip_bf16.h>
using bf16 = __hip_bfloat16;
#endif
#else
#include <cuda_bf16.h>
using bf16 = __nv_bfloat16;
#endif
