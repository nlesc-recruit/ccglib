#if defined(__HIP__)
#include <hip/hip_fp8.h>
using fp8_e4m3 = __hip_fp8_e4m3;
#else
#include <cuda_fp8.h>
using fp8_e4m3 = __nv_fp8_e4m3;
#endif
