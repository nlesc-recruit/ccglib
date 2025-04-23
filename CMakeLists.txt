cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(
  ccglib
  DESCRIPTION "Complex CUDA GEMM"
  VERSION 0.1
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

if(NOT DEFINED CCGLIB_BACKEND)
  set(CCGLIB_BACKEND "CUDA")
endif()
set(CCGLIB_BACKEND
    ${CCGLIB_BACKEND}
    CACHE STRING "GPU backend API to use")
set_property(CACHE CCGLIB_BACKEND PROPERTY STRINGS "CUDA" "HIP")

if(${CCGLIB_BACKEND} STREQUAL "CUDA")
  set(CCGLIB_BACKEND_CUDA True)
elseif(${CCGLIB_BACKEND} STREQUAL "HIP")
  set(CCGLIB_BACKEND_HIP True)
else()
  message(FATAL_ERROR "Invalid value for CCGLIB_BACKEND: ${CCGLIB_BACKEND}")
endif()

find_package(OpenMP REQUIRED)

set(CUDAWRAPPERS_BACKEND ${CCGLIB_BACKEND})
if(${CCGLIB_BACKEND_HIP})
  # Workaround missing omp.h for compilation of sources with language property
  # HIP by adding the OpenMP include directory of the host compiler to the
  # OpenMP target.
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    get_filename_component(OpenMP_LIB_DIR ${OpenMP_gomp_LIBRARY} DIRECTORY)
    find_path(
      OpenMP_INCLUDE_DIR omp.h
      PATHS ${OpenMP_LIB_DIR} "${OpenMP_LIB_DIR}/.."
      PATH_SUFFIXES
        "include"
        "lib/gcc/x86_64-pc-linux-gnu/${CMAKE_CXX_COMPILER_VERSION}/include")
    target_include_directories(OpenMP::OpenMP_CXX SYSTEM
                               INTERFACE ${OpenMP_INCLUDE_DIR})
  endif()

  enable_language(HIP)
  set(LINK_gpu_runtime hip::host)
elseif(${CCGLIB_BACKEND_CUDA})
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(LINK_gpu_runtime CUDA::cudart)
endif()

# fetch cudawrappers
include(FetchContent)
FetchContent_Declare(
  cudawrappers
  GIT_REPOSITORY https://github.com/nlesc-recruit/cudawrappers
  GIT_TAG "main")
FetchContent_MakeAvailable(cudawrappers)

include(cmake/xtensor.cmake)

option(CCGLIB_BUILD_TESTING "Build the test suite" OFF)
option(CCGLIB_BUILD_BENCHMARK "Build the benchmark suite" OFF)
option(CCGLIB_BENCHMARK_WITH_PMT
       "Enable Power Measurement Toolkit support in the benchmark suite" OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(src)

if(CCGLIB_BUILD_TESTING)
  enable_testing()
  add_subdirectory(test)
endif()

if(CCGLIB_BUILD_BENCHMARK)
  add_subdirectory(benchmark)
endif()
