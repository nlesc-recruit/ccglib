#ifndef SYNC_COPIES_H_
#define SYNC_COPIES_H_

// Helper function to perform a synchronous memory copy
// T: Data type for the copy operation
// NBYTES: Total number of bytes to copy
// NTHREADS: Number of threads participating in the copy
template <typename T, size_t NBYTES, size_t NTHREADS>
__device__ void copy_sync(void *dest, const void *src, size_t tid) {
  static_assert(NBYTES % sizeof(T) == 0);

  // Calculate the number of elements of type T to copy
  const size_t n = NBYTES / sizeof(T);

  // Get the destination and source pointers of type T
  T *dest_ptr = reinterpret_cast<T *>(dest);
  const T *src_ptr = reinterpret_cast<const T *>(src);

// Distribute the ynchronous copy over the threads
#pragma unroll
  for (size_t idx = tid; idx < n; idx += NTHREADS) {
    dest_ptr[idx] = src_ptr[idx];
  }
}

#endif // SYNC_COPIES_H_