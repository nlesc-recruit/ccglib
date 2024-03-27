#include <cooperative_groups/memcpy_async.h>
namespace cg = cooperative_groups;

// device function for async copy from gmem to smem
template <unsigned NBYTES, size_t NTHREADS>
__device__ void copy_async(void *dest, const void *src,
                           nvcuda::experimental::pipeline &pipe, unsigned tid) {
  int bytes_to_go = (NBYTES + NTHREADS - 1) / NTHREADS;

#pragma unroll
  while (bytes_to_go >= sizeof(int4)) {
    memcpy_async(((int4 *)dest)[tid], ((const int4 *)src)[tid], pipe);
    bytes_to_go -= sizeof(int4);
    src = (void *)((int4 *)src + NTHREADS);
    dest = (void *)((int4 *)dest + NTHREADS);
  }

  if (bytes_to_go >= sizeof(int2)) {
    memcpy_async(((int2 *)dest)[tid], ((const int2 *)src)[tid], pipe);
    bytes_to_go -= sizeof(int2);
    src = (void *)((int2 *)src + NTHREADS);
    dest = (void *)((int2 *)dest + NTHREADS);
  }

  if (bytes_to_go >= sizeof(int)) {
    memcpy_async(((int *)dest)[tid], ((const int *)src)[tid], pipe);
    bytes_to_go -= sizeof(int);
    src = (void *)((int *)src + NTHREADS);
    dest = (void *)((int *)dest + NTHREADS);
  }

  if (bytes_to_go >= sizeof(short)) {
    memcpy_async(((short *)dest)[tid], ((const short *)src)[tid], pipe);
    bytes_to_go -= sizeof(short);
    src = (void *)((short *)src + NTHREADS);
    dest = (void *)((short *)dest + NTHREADS);
  }

  if (bytes_to_go >= sizeof(char)) {
    memcpy_async(((char *)dest)[tid], ((const char *)src)[tid], pipe);
    bytes_to_go -= sizeof(char);
    src = (void *)((char *)src + NTHREADS);
    dest = (void *)((char *)dest + NTHREADS);
  }
}