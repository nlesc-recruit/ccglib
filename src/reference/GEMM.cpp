#include <atomic>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <xtensor/xadapt.hpp>
#include <xtensor/xcomplex.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

#include <ccglib/common/helper.h>
#include <ccglib/gemm/reference.h>

namespace {
template <typename Tin, typename Tout>
void Run(const Tin *a, const Tin *b, Tout *c, size_t M, size_t N, size_t K,
         ccglib::mma::MemOrder output_mem_order) {

// FIXME: When using 'half' output type on an AMD GPU, the computation cannot be
// done using 'half' type. This is because the required operators, `__half&
// operator+(const __half& x)` and `__half& operator*(const __half& x)`, are
// missing. According to the HIP documentation, they should be available. A bug
// report has been submitted: https://github.com/ROCm/HIP/issues/3690 but the
// fix has not yet been upstreamed.
#if defined(__HIP_PLATFORM_AMD__)
  using ComputeType =
      typename std::conditional<std::is_same<Tin, half>::value ||
                                    std::is_same<Tout, half>::value,
                                float, Tout>::type;
#else
  using ComputeType = Tout;
#endif

  const std::array<size_t, 3> a_shape = {2, M, K};
  const std::array<size_t, 3> b_shape = {2, N, K};
  std::array<size_t, 3> c_shape;
  if (output_mem_order == ccglib::mma::row_major) {
    c_shape = {2, M, N};
  } else {
    c_shape = {2, N, M};
  }

  const size_t a_size = 2 * M * K;
  const size_t b_size = 2 * N * K;
  const size_t c_size = 2 * M * N;

  auto a_view = xt::adapt(a, a_size, xt::no_ownership(), a_shape);
  auto b_view = xt::adapt(b, b_size, xt::no_ownership(), b_shape);
  auto c_view = xt::adapt(c, c_size, xt::no_ownership(), c_shape);

#pragma omp parallel for collapse(2)
  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      ComputeType sum_real = 0;
      ComputeType sum_imag = 0;

      for (size_t k = 0; k < K; ++k) {
        const ComputeType a_real = a_view(0, m, k);
        const ComputeType a_imag = a_view(1, m, k);
        const ComputeType b_real = b_view(0, n, k);
        const ComputeType b_imag = b_view(1, n, k);

        ComputeType term_real = a_real * b_real - a_imag * b_imag;
        ComputeType term_imag = a_real * b_imag + a_imag * b_real;

        sum_real += term_real;
        sum_imag += term_imag;
      }

      if (output_mem_order == ccglib::mma::row_major) {
        c_view(0, m, n) = sum_real;
        c_view(1, m, n) = sum_imag;
      } else {
        c_view(0, n, m) = sum_real;
        c_view(1, n, m) = sum_imag;
      }
    }
  }
}

template <typename T> inline T bitwise_xor(const T &a, const T &b) {
  return a ^ b;
}

template <typename T> inline int popcount(T x) {
  int count = 0;
  while (x) {
    count += x & 1;
    x >>= 1;
  }
  return count;
}

void run_binary(const unsigned *a, const unsigned *b, int *c, size_t M,
                size_t N, size_t K, ccglib::mma::MemOrder output_mem_order) {
  // M, N, K are the number of 1-bit samples
  // the actual shape of the input data is different along the fastest changing
  // axis (=K) because values are packed into unsigned ints
  const size_t samples_per_element = sizeof(unsigned) * CHAR_BIT;
  const size_t K_PACKED = ccglib::helper::ceildiv(K, samples_per_element);
  const size_t K_PADDED = K_PACKED * samples_per_element;
  const size_t K_PADDING = K_PADDED - K;

  const std::array<size_t, 3> a_shape = {2, M, K_PACKED};
  const std::array<size_t, 3> b_shape = {2, N, K_PACKED};
  std::array<size_t, 3> c_shape;
  if (output_mem_order == ccglib::mma::row_major) {
    c_shape = {2, M, N};
  } else {
    c_shape = {2, N, M};
  }

  const size_t a_size = 2 * M * K_PACKED;
  const size_t b_size = 2 * N * K_PACKED;
  const size_t c_size = 2 * M * N;

  auto a_view = xt::adapt(a, a_size, xt::no_ownership(), a_shape);
  auto b_view = xt::adapt(b, b_size, xt::no_ownership(), b_shape);
  auto c_view = xt::adapt(c, c_size, xt::no_ownership(), c_shape);

#pragma omp parallel for collapse(2)
  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      /*
      The dot product of two vectors of 1-bit values can be understood as
      follows: each value is 1 or -1:
      a  | b  | product
      -1 | -1 |  1
       1 | -1 | -1
      -1 |  1 | -1
       1 |  1 |  1

      In the input data, -1 is represented by binary 0, and 1 is represented by
      binary 1. The product operation is then equivalent to XOR, with -1
      represented by binary 1, and 1 by binary 0, exactly opposite to the input.
      For the accumulation we count the number of bits set to one, i.e. the
      popcount operation. Each bit set to 1 represents -1, so if we sum K values
      we calculate the final result as:
      (number of bits set to 1) * -1 + (number of bits set to 0) * 1 =
      (popcount output) * -1 + (K - popcount output) * 1 =
      K - 2 * (popcount output) =
      K - 2 * popc(a xor b)

      This is typically how this operation is executed on the tensor cores,
      hence why we do it the same way in this reference implementation.

      Because we are doing a complex matrix-matrix multiplication, actually 2K
      values are summed together so the output is 2 * (K - popc(a xor b))

      Lastly, we need to consider padding. The 1-bit values are packed into
      32-bit unsigned ints, but the number of input samples may not be a
      multiple of 32. The remaining bits are set to binary zero, i.e. -1.

      For the real part of the output, two MMA results are subtracted from each
      other and the padding effect cancels out. For the imaginary part, we twice
      include -1 * -1 * (number of padding samples), i.e. we need to subtract
      twice the amount of padded samples from the result. Given complex
      multiplication:
      (ar + ai i) * (br + bi i) ->
      sum_real = ar * br - ai * bi
      sum_imag = ar * bi + ai * br

      We arrive at the following steps for the complete computation, where K is
      the number of padded samples and instead of subtraction (which the tensor
      cores do no support) we negate one of the inputs:
      sum_real = popc(ar xor br) + popc(ai xor ~bi)
      sum_imag = popc(ar xor bi) + popc(ai xor br)
      sum_real = 2 * (K - sum_real)
      sum_imag = 2 * (K - padding - sum_imag)
      */

      int sum_real = 0;
      int sum_imag = 0;
      for (size_t k = 0; k < K_PACKED; ++k) {
        const unsigned a_real = a_view(0, m, k);
        const unsigned a_imag = a_view(1, m, k);
        const unsigned b_real = b_view(0, n, k);
        const unsigned b_imag = b_view(1, n, k);

        sum_real += popcount(bitwise_xor(a_real, b_real)) +
                    popcount(bitwise_xor(a_imag, ~b_imag));
        sum_imag += popcount(bitwise_xor(a_real, b_imag)) +
                    popcount(bitwise_xor(a_imag, b_real));
      }

      if (output_mem_order == ccglib::mma::row_major) {
        c_view(0, m, n) = 2 * (K_PADDED - sum_real);
        c_view(1, m, n) = 2 * (K_PADDED - K_PADDING - sum_imag);
      } else {
        c_view(0, n, m) = 2 * (K_PADDED - sum_real);
        c_view(1, n, m) = 2 * (K_PADDED - K_PADDING - sum_imag);
      }
    }
  }
}
} // namespace

namespace ccglib::reference {

void GEMM::Run(const half *a, const half *b, half *c, size_t M, size_t N,
               size_t K, ccglib::mma::MemOrder output_mem_order) {
  ::Run<half, half>(a, b, c, M, N, K, output_mem_order);
}

void GEMM::Run(const half *a, const half *b, float *c, size_t M, size_t N,
               size_t K, ccglib::mma::MemOrder output_mem_order) {
  ::Run<half, float>(a, b, c, M, N, K, output_mem_order);
}

void GEMM::Run(const float *a, const float *b, half *c, size_t M, size_t N,
               size_t K, ccglib::mma::MemOrder output_mem_order) {
  ::Run<float, half>(a, b, c, M, N, K, output_mem_order);
}

void GEMM::Run(const float *a, const float *b, float *c, size_t M, size_t N,
               size_t K, ccglib::mma::MemOrder output_mem_order) {
  ::Run<float, float>(a, b, c, M, N, K, output_mem_order);
}

void GEMM::Run(const unsigned *a, const unsigned *b, int *c, size_t M, size_t N,
               size_t K, ccglib::mma::MemOrder output_mem_order) {
  ::run_binary(a, b, c, M, N, K, output_mem_order);
}

} // end namespace ccglib::reference
