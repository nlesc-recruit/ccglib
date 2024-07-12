extern "C" __global__ void
pack_bits(unsigned *output, const unsigned char *input, const size_t n) {
  size_t tid = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
  if (tid >= n) {
    return;
  }

  unsigned output_value = __ballot_sync(__activemask(), input[tid]);
  if (tid % 32 == 0) {
    output[tid / 32] = output_value;
  }
}

extern "C" __global__ void unpack_bits(unsigned char *output,
                                       const unsigned *input, const size_t n) {
  size_t tid = threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
  if (tid >= n) {
    return;
  }

  unsigned value = input[tid / 32];
  output[tid] = (value >> (tid % 32)) & 1;
}
