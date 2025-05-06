#if defined(__HIP__)
#include <hip/hip_bf16.h>
using bf16 = __hip_bfloat16;
#else
#include <cuda_bf16.h>
using bf16 = __nv_bfloat16;
#endif
