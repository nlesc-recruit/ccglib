# ccglib
[![CI status linting](https://img.shields.io/github/actions/workflow/status/nlesc-recruit/ccglib/linting.yml?label=linting)](https://github.com/nlesc-recruit/ccglib/actions/workflows/linting.yml)
[![CI status cuda](https://img.shields.io/github/actions/workflow/status/nlesc-recruit/ccglib/test_cuda.yml?label=test%20CUDA%20%28A4000%29)](https://github.com/nlesc-recruit/ccglib/actions/workflows/test_cuda.yml)
[![CI status hip](https://img.shields.io/github/actions/workflow/status/nlesc-recruit/ccglib/test_hip.yml?label=test%20HIP%20%28W7700%29)](https://github.com/nlesc-recruit/ccglib/actions/workflows/test_hip.yml)

The Complex Common GEMM Library (ccglib) provides a simple C++ interface for complex-valued matrix multiplication on GPU tensor and matrix cores, supporting both CUDA and HIP.

## Requirements

- NVIDIA: Any GPU with tensor cores and support for asynchronous memory copies, i.e. Ampere generation or newer.
- AMD: Any GPU with matrix cores, i.e. CDNA1 or newer, RDNA3 or newer.

Software | Minimum version
-------- | ---------------
CUDA     | 11.0
ROCm     | 6.1
CMake    | 3.20

Note: Certain input/output types are only supported by specific GPU architectures, see the [table below](#supported-data-types-and-matrix-layouts) for details.

## Installation
CMake is used to build ccglib. It can either be built as a library, or used in another project as external dependency through CMake.
To build ccglib locally, run:
```shell
git clone https://git.astron.nl/RD/recruit/ccglib
cd ccglib
cmake -S . -B build
make -C build
```

To use ccglib as an external dependency, add the following to the `CMakeLists.txt` file of your project:
```cmake
include(FetchContent)

FetchContent_Declare(
  ccglib
  GIT_REPOSITORY https://git.astron.nl/RD/recruit/ccglib
  GIT_TAG main)
FetchContent_MakeAvailable(ccglib)
```

Then link ccglib into your executable or library with:
```cmake
target_link_libraries(<your_target> ccglib)
```

The following build options are available:
Option                      | Description | Default
------                      | ----------- | -------
`CCGLIB_BACKEND`            | GPU backend API to use, either `CUDA` or `HIP`                                                  | `CUDA`
`CCGLIB_BUILD_TESTING`      | Build the test suite. In HIP mode, it may be required to use `hipcc` as the host compiler.      | `OFF`
`CCGLIB_BUILD_BENCHMARK`    | Build the benchmark suite                                                                       | `OFF`
`CCGLIB_BENCHMARK_WITH_PMT` | Enable [Power Measurement Toolkit](https://git.astron.nl/RD/pmt) support in the benchmark suite | `OFF`

## Supported data types and matrix layouts
ccglib supports a range of input/output data types, depending on the available hardware:

Input type  | Output type      | NVIDIA | AMD | Notes
----------  | -----------      | ------ | - | -
float16     | float32/float16\* | ✅              | ✅ | -
float32     | float32/float16\* | ❌              | CDNA only  | | -
tensorfloat | float32/float16\* | Ampere or newer | ❌ | Input data must be in float32 format, conversion to tensorfloat is automatic
int1        | int32            | ✅              | ❌ | Input bits must be packed into int32 values. ccglib provides a tool to do this

\* float16 output is native float32 output downcasted to float16.


With matrix-matrix multiplication defined as `C = A x B`, ccglib requires the A matrix to be in row-major format and the B matrix to be in column-major format. The C matrix can be either row-major or column-major.

The real and imaginary samples can either be interleaved (i.e. the complex axis is the fastest changing axis) or planar (i.e. the complex axis is the slowest changing axis of a single matrix).

Two variants of the GEMM are provided: a basic and an optimized version. The basic GEMM requires the input to be in planar format and the output is planar as well. The optimized GEMM has a complicated input format. A transpose operation is provided to convert input matrices of either interleaved or planar format to the format required by the optimized GEMM. The output can be either planar or interleaved, with planar providing the best performance.

ccglib supports running multiple GEMM operations at once using a batch size parameter. The matrices must be stored contiguous in device memory. The output will be a set of matrices contiguous in memory as well.

As as example, consider a row-major A matrix of `M` rows and `K` colums, a column-major B matrix of `K` rows and `N` columns, and a resulting row-major C matrix of `M` rows and `N` columns. With planar complex samples, the shapes of the matrices for a basic GEMM are as follows:
- A: `BATCH x COMPLEX x M x K`
- B: `BATCH x COMPLEX x N x K`
- C: `BATCH x COMPLEX x M x N`


## Example usage
Refer to the examples folder for typical usage examples.

ccglib uses [cudawrappers](https://github.com/nlesc-recruit/cudawrappers) to provide a unified interface to CUDA and HIP. Refer to the [cudawrappers documentation](https://cudawrappers.readthedocs.io/en/latest/) for more details.
