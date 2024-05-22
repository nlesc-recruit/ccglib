#ifndef ASYNC_COPIES_H_
#define ASYNC_COPIES_H_

#include <cuda/pipeline>

// device function for async copy from gmem to smem
template <unsigned NBYTES, size_t NTHREADS>
__device__ void copy_async(void *dest, const void *src,
                           cuda::pipeline<cuda::thread_scope_thread> &pipe,
                           unsigned tid) {
  int bytes_to_go = (NBYTES + NTHREADS - 1) / NTHREADS;

#pragma unroll
  while (bytes_to_go >= sizeof(int4)) {
    const auto shape = cuda::aligned_size_t<alignof(int4)>(sizeof(int4));
    cuda::memcpy_async(&((int4 *)dest)[tid], &((const int4 *)src)[tid], shape,
                       pipe);
    bytes_to_go -= sizeof(int4);
    src = (void *)((int4 *)src + NTHREADS);
    dest = (void *)((int4 *)dest + NTHREADS);
  }

  if (bytes_to_go >= sizeof(int2)) {
    const auto shape = cuda::aligned_size_t<alignof(int2)>(sizeof(int2));
    cuda::memcpy_async(&((int2 *)dest)[tid], &((const int2 *)src)[tid], shape,
                       pipe);
    bytes_to_go -= sizeof(int2);
    src = (void *)((int2 *)src + NTHREADS);
    dest = (void *)((int2 *)dest + NTHREADS);
  }

  if (bytes_to_go >= sizeof(int)) {
    const auto shape = cuda::aligned_size_t<alignof(int)>(sizeof(int));
    cuda::memcpy_async(&((int *)dest)[tid], &((const int *)src)[tid], shape,
                       pipe);
    bytes_to_go -= sizeof(int);
    src = (void *)((int *)src + NTHREADS);
    dest = (void *)((int *)dest + NTHREADS);
  }

  if (bytes_to_go >= sizeof(short)) {
    const auto shape = cuda::aligned_size_t<alignof(short)>(sizeof(short));
    cuda::memcpy_async(&((short *)dest)[tid], &((const short *)src)[tid], shape,
                       pipe);
    bytes_to_go -= sizeof(short);
    src = (void *)((short *)src + NTHREADS);
    dest = (void *)((short *)dest + NTHREADS);
  }

  if (bytes_to_go >= sizeof(char)) {
    const auto shape = cuda::aligned_size_t<alignof(char)>(sizeof(char));
    cuda::memcpy_async(&((char *)dest)[tid], &((const char *)src)[tid], shape,
                       pipe);
    bytes_to_go -= sizeof(char);
    src = (void *)((char *)src + NTHREADS);
    dest = (void *)((char *)dest + NTHREADS);
  }
}

#endif // ASYNC_COPIES_H_