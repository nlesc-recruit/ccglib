#if defined(__HIP__)
#include <limits.h>
#else
#include <cuda/std/limits>
#endif

extern "C" __global__ void pack_bits(unsigned *output,
                                     const unsigned char *input,
                                     bool input_complex_last) {
  size_t tid = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
  if (tid >= N) {
    return;
  }

  size_t input_index = tid;
  if (input_complex_last) {
    // map from real0, real1, .... imag0, imag1... indexing to
    // real0, imag0, real1, imag1, ...
    input_index *= 2;
    if (input_index >= N) {
      input_index -= N - 1;
    }
  }

#if WARP_SIZE == 32
  unsigned output_value = __ballot_sync(__activemask(), input[input_index]);
  if (tid % WARP_SIZE == 0) {
    // ensure to keep only warp_size bits as HIP always returns a 64-bit value
    output[tid / WARP_SIZE] = output_value & 0xFFFFFFFF;
  }
#elif WARP_SIZE == 64
  unsigned long output_value =
      __ballot_sync(__activemask(), input[input_index]);
  if (tid % WARP_SIZE == 0) {
    // because output is 32-bit type, the last thread might try to write 32 bits
    // beyond the end of the memory allocation if we write per 64 bits. Avoid
    // this by explicitly checking and only writing the lower 32 bits in this
    // case.
    const size_t index = 2 * (tid / WARP_SIZE);
    const unsigned nr_bits = sizeof(unsigned) * CHAR_BIT;
    if (index == (N / nr_bits - 1)) {
      output[index] = output_value & 0xFFFFFFFF;
    } else {
      reinterpret_cast<unsigned long *>(output)[tid / WARP_SIZE] = output_value;
    }
  }
#else
#error WARP_SIZE must be 32 or 64
#endif
}

extern "C" __global__ void unpack_bits(unsigned char *output,
                                       const unsigned *input) {
  size_t tid = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
  if (tid >= N) {
    return;
  }

  const unsigned nr_bits = sizeof(unsigned) * CHAR_BIT;

  unsigned value = input[tid / nr_bits];
  output[tid] = (value >> (tid % nr_bits)) & 1;
}
