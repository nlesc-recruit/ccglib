#if defined(__HIP__) && !defined(__HIP_PLATFORM_NVIDIA__)
#error "Packing kernel is only available for NVIDIA GPUs"
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

  unsigned output_value = __ballot_sync(__activemask(), input[input_index]);
  if (tid % 32 == 0) {
    output[tid / 32] = output_value;
  }
}

extern "C" __global__ void unpack_bits(unsigned char *output,
                                       const unsigned *input) {
  size_t tid = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
  if (tid >= N) {
    return;
  }

  unsigned value = input[tid / 32];
  output[tid] = (value >> (tid % 32)) & 1;
}
