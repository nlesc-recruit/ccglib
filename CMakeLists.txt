cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(
  ccglib
  DESCRIPTION "Complex CUDA GEMM"
  VERSION 0.1
  LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}")

# fetch cudawrappers
include(FetchContent)
FetchContent_Declare(
  cudawrappers
  GIT_REPOSITORY https://github.com/nlesc-recruit/cudawrappers
  GIT_TAG main)
FetchContent_MakeAvailable(cudawrappers)

include(cmake/xtensor.cmake)

option(CCGLIB_BUILD_TESTING "Build the test suite" OFF)

add_subdirectory(src/reference)
add_subdirectory(src/transpose)
add_subdirectory(src/mma)

if(CCGLIB_BUILD_TESTING)
  enable_testing()
  add_subdirectory(test)
endif()
