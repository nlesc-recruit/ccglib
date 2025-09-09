#ifndef TYPE_SELECTOR_H_
#define TYPE_SELECTOR_H_

#if defined(__HIP_PLATFORM_AMD__)
#include <limits.h>
#include <rocwmma/rocwmma.hpp>
namespace wmma = rocwmma;
#else
#include <cuda/pipeline>
#include <mma.h>
using namespace nvcuda;
#endif

#include "ccglib/bf16.h"
#include "ccglib/fp16.h"

#include "value_type.h"
using ccglib::ValueType;

#ifndef COMPLEX
#define COMPLEX 2
#endif
#ifndef REAL
#define REAL 0
#endif
#ifndef IMAG
#define IMAG 1
#endif

// All values related to data layout must be defined at compile time
#if !defined BATCH_SIZE || !defined M_GLOBAL || !defined N_GLOBAL ||           \
    !defined K_GLOBAL
#error                                                                         \
    "BATCH_SIZE, M_GLOBAL, N_GLOBAL, and K_GLOBAL values per block, warp, tensor core must be defined at compile time"
#endif

// Check memory layout of A and B matrix
#ifdef A_COL_MAJOR
#error "float GEMM does not currently support col-major A matrix"
#endif
#ifdef B_ROW_MAJOR
#error "float GEMM does not currently support row-major B matrix"
#endif

#if defined(__HIP_PLATFORM_AMD__) && (NBIT_IN == 1)
#error "1-bit GEMM is only available for NVIDIA GPUs"
#endif

// The kernel assumes the dimensions of the input matrices are a multiple
// of M_PER_BLOCK and/or N_PER_BLOCK and/or K_PER_WMMA. The *_GLOBAL_PADDED
// constants are padded to be multiples.
#define M_IS_PADDED ((M_GLOBAL % M_PER_BLOCK) != 0)
#define N_IS_PADDED ((N_GLOBAL % N_PER_BLOCK) != 0)
#define K_IS_PADDED ((K_GLOBAL % K_PER_WMMA) != 0)

#define M_GLOBAL_PADDED ((M_GLOBAL / M_PER_BLOCK + M_IS_PADDED) * M_PER_BLOCK)
#define N_GLOBAL_PADDED ((N_GLOBAL / N_PER_BLOCK + N_IS_PADDED) * N_PER_BLOCK)
#define K_GLOBAL_PADDED ((K_GLOBAL / K_PER_WMMA + K_IS_PADDED) * K_PER_WMMA)

// The TypeSelector struct is used to determine the size and type of data
// structures used in the GEMM kernel. It relies on template deduction based on
// the kernel input and output data types, and is specialized below for the
// different supported input and output types.
template <ValueType IN, ValueType OUT> struct TypeSelector {
  static_assert((IN == ValueType::int1 && OUT == ValueType::int32) ||
                    ((IN == ValueType::float16 || IN == ValueType::float32) &&
                     (OUT == ValueType::float16 || ValueType::float32)),
                "Unsupported combination of input/output data types");
};

template <> struct TypeSelector<ValueType::int1, ValueType::int32> {
  using Tin = unsigned int;
#if defined(__HIP_PLATFORM_AMD__) || (__CUDA_ARCH__ < 730)
  using Ttc = void;
#else
  using Ttc = wmma::experimental::precision::b1;
#endif
  using Tshared = int;
  using Tout = int;

  static constexpr unsigned PACKING_FACTOR = (sizeof(Tin) * CHAR_BIT / NBIT_IN);
  static constexpr size_t OVERRIDE_K_PER_WMMA = 0;
  static constexpr bool IS_DOWNCAST_OP = false;
};

template <> struct TypeSelector<ValueType::float8e4m3, ValueType::float32> {
  using Tin = __nv_fp8_e4m3;
  using Ttc = __nv_fp8_e4m3;
  using Tshared = float;
  using Tout = float;

  static constexpr unsigned PACKING_FACTOR = 1;
  static constexpr size_t OVERRIDE_K_PER_WMMA = 0;
  static constexpr bool IS_DOWNCAST_OP = false;
};

template <> struct TypeSelector<ValueType::bfloat16, ValueType::bfloat16> {
  using Tin = bf16;
  using Ttc = bf16;
  using Tshared = bf16;
  using Tout = bf16;

  static constexpr unsigned PACKING_FACTOR = 1;
  static constexpr size_t OVERRIDE_K_PER_WMMA = 0;
  static constexpr bool IS_DOWNCAST_OP = false;
};

template <> struct TypeSelector<ValueType::bfloat16, ValueType::float32> {
  using Tin = bf16;
  using Ttc = bf16;
  using Tshared = float;
  using Tout = float;

  static constexpr unsigned PACKING_FACTOR = 1;
  static constexpr size_t OVERRIDE_K_PER_WMMA = 0;
  static constexpr bool IS_DOWNCAST_OP = false;
};

template <> struct TypeSelector<ValueType::float16, ValueType::float16> {
  using Tin = half;
  using Ttc = half;
  using Tshared = half;
  using Tout = half;

  static constexpr unsigned PACKING_FACTOR = 1;
  static constexpr size_t OVERRIDE_K_PER_WMMA = 0;
  static constexpr bool IS_DOWNCAST_OP = false;
};

template <> struct TypeSelector<ValueType::float16, ValueType::float32> {
  using Tin = half;
  using Ttc = half;
  using Tshared = float;
  using Tout = float;

  static constexpr unsigned PACKING_FACTOR = 1;
  static constexpr size_t OVERRIDE_K_PER_WMMA = 0;
  static constexpr bool IS_DOWNCAST_OP = false;
};

template <> struct TypeSelector<ValueType::float32, ValueType::float32> {
  using Tin = float;
#if defined(__HIP_PLATFORM_AMD__)
  using Ttc = float;
#elif (__CUDA_ARCH__ >= 800)
  using Ttc = wmma::precision::tf32;
#else
  using Ttc = void;
#endif
  using Tshared = float;
  using Tout = float;

  static constexpr unsigned PACKING_FACTOR = 1;
  static constexpr size_t OVERRIDE_K_PER_WMMA = 0;
  static constexpr bool IS_DOWNCAST_OP = false;
};

template <> struct TypeSelector<ValueType::float32, ValueType::bfloat16> {
  using Tin = float;
#ifdef __HIP_PLATFORM_AMD__
  using Ttc = float;
#elif (__CUDA_ARCH__ >= 800)
  using Ttc = wmma::precision::tf32;
#else
  using Ttc = void;
#endif
  using Tshared = float;
  using Tout = bf16;

  static constexpr unsigned PACKING_FACTOR = 1;
  static constexpr size_t OVERRIDE_K_PER_WMMA = 8;
  static constexpr bool IS_DOWNCAST_OP = true;
};

template <> struct TypeSelector<ValueType::float32, ValueType::float16> {
  using Tin = float;
#if defined(__HIP_PLATFORM_AMD__)
  using Ttc = float;
#elif (__CUDA_ARCH__ >= 800)
  using Ttc = wmma::precision::tf32;
#else
  using Ttc = void;
#endif
  using Tshared = float;
  using Tout = half;

  static constexpr unsigned PACKING_FACTOR = 1;
  static constexpr size_t OVERRIDE_K_PER_WMMA = 8;
  static constexpr bool IS_DOWNCAST_OP = true;
};

// Create aliases for the defined types
using DeviceTraits = TypeSelector<static_cast<ValueType>(TYPE_IN),
                                  static_cast<ValueType>(TYPE_OUT)>;

using Tin = typename DeviceTraits::Tin;
using Ttc = typename DeviceTraits::Ttc;
using Tshared = typename DeviceTraits::Tshared;
using Tout = typename DeviceTraits::Tout;

// basic data layout
using A_t =
    Tin[BATCH_SIZE][COMPLEX][M_GLOBAL][K_GLOBAL / DeviceTraits::PACKING_FACTOR];
using B_t =
    Tin[BATCH_SIZE][COMPLEX][N_GLOBAL][K_GLOBAL / DeviceTraits::PACKING_FACTOR];

#if defined(C_COMPLEX_PLANAR)
#ifdef C_ROW_MAJOR
using C_t = Tout[BATCH_SIZE][COMPLEX][M_GLOBAL][N_GLOBAL];
#else
using C_t = Tout[BATCH_SIZE][COMPLEX][N_GLOBAL][M_GLOBAL];
#endif
#elif defined(C_COMPLEX_INTERLEAVED)
#ifdef C_ROW_MAJOR
using C_t = Tout[BATCH_SIZE][M_GLOBAL][N_GLOBAL][COMPLEX];
#else
using C_t = Tout[BATCH_SIZE][N_GLOBAL][M_GLOBAL][COMPLEX];
#endif
#endif

constexpr size_t A_s_size =
    NBUFFER * COMPLEX * M_PER_BLOCK * K_PER_WMMA / DeviceTraits::PACKING_FACTOR;
constexpr size_t B_s_size =
    NBUFFER * COMPLEX * N_PER_BLOCK * K_PER_WMMA / DeviceTraits::PACKING_FACTOR;

constexpr size_t C_s_size = (M_PER_BLOCK / M_PER_WARP) *
                            (N_PER_BLOCK / N_PER_WARP) * M_PER_WMMA *
                            N_PER_WMMA;
static_assert((A_s_size + B_s_size) * sizeof(Tin) >= C_s_size * sizeof(Tout),
              "A_s + B_s >= C_s");

static constexpr size_t ACCUMULATOR_K_PER_WMMA =
    (DeviceTraits::OVERRIDE_K_PER_WMMA > 0) ? DeviceTraits::OVERRIDE_K_PER_WMMA
                                            : K_PER_WMMA;

using Accumulator_t =
    typename wmma::fragment<wmma::accumulator, M_PER_WMMA, N_PER_WMMA,
                            ACCUMULATOR_K_PER_WMMA, Tshared>;

#if NBIT_OUT < NBIT_IN
#define REQUIRES_DOWNCAST 1
#else
#define REQUIRES_DOWNCAST 0
#endif

#define REQUIRES_SHARED_MEMORY                                                 \
  (M_IS_PADDED || N_IS_PADDED || C_COMPLEX_INTERLEAVED || REQUIRES_DOWNCAST)

#endif // TYPE_SELECTOR_H_
