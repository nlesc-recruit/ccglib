#ifndef ASYNC_COPIES_H_
#define ASYNC_COPIES_H_

#include <cuda/pipeline>

// Helper function to perform an asynchronous memory copy
// T: Data type for the copy operation
// NBYTES: Total number of bytes to copy
// NTHREADS: Number of threads participating in the copy
template <typename T, size_t NBYTES, size_t NTHREADS>
inline __device__ void
copy_async(void *dest, const void *src, size_t *offset,
           cuda::pipeline<cuda::thread_scope_thread> &pipe, size_t tid) {
  const auto shape = cuda::aligned_size_t<alignof(T)>(sizeof(T));

  // Calculate the number of elements of type T to copy
  const size_t n = (NBYTES - *offset) / sizeof(T);

  // Get the destination and source pointers of type T at the current offset
  T *dest_ptr = reinterpret_cast<T *>(reinterpret_cast<char *>(dest) + *offset);
  const T *src_ptr = reinterpret_cast<const T *>(
      reinterpret_cast<const char *>(src) + *offset);

// Distribute the asynchronous copy over the threads
#pragma unroll
  for (size_t i = tid; i < n; i += NTHREADS) {
    cuda::memcpy_async(&(dest_ptr[i]), &(src_ptr[i]), shape, pipe);
  }

  *offset += n * sizeof(T);
}

// Device function to perform an asynchronous memory copy
// NBYTES: Total number of bytes to copy
// NTHREADS: Number of threads participating in the copy
template <size_t NBYTES, size_t NTHREADS>
__device__ void copy_async(void *dest, const void *src,
                           cuda::pipeline<cuda::thread_scope_thread> &pipe,
                           size_t tid) {
  size_t offset = 0;

  // Perform the copy operation for various data types
  copy_async<int4, NBYTES, NTHREADS>(dest, src, &offset, pipe, tid);
  copy_async<int2, NBYTES, NTHREADS>(dest, src, &offset, pipe, tid);
  copy_async<int, NBYTES, NTHREADS>(dest, src, &offset, pipe, tid);
  copy_async<short, NBYTES, NTHREADS>(dest, src, &offset, pipe, tid);
  copy_async<char, NBYTES, NTHREADS>(dest, src, &offset, pipe, tid);
}

#endif // ASYNC_COPIES_H_