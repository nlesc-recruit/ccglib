cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

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

include(cmake/helper.cmake)

option(CCGLIB_BUILD_TESTING "Build the test suite" OFF)

if(CCGLIB_BUILD_TESTING)
  enable_testing()
endif()

add_subdirectory(src/reference)
add_subdirectory(src/transpose)
add_subdirectory(src/mma)

# build beamformer
add_executable(bf src/bf.cu)
target_include_directories(bf PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
                                      ${CMAKE_SOURCE_DIR}/kernels)
target_link_libraries(
  bf
  PRIVATE cudawrappers::cu
          cudawrappers::nvrtc
          OpenMP::OpenMP_CXX
          xtensor
          $<TARGET_OBJECTS:gemm-reference>
          $<TARGET_OBJECTS:helper>
          $<TARGET_OBJECTS:gemm-mma>)
target_embed_source(bf kernels/gemm_kernel.cu)
target_embed_source(bf kernels/transpose_kernel.cu)
set_target_properties(bf PROPERTIES CUDA_ARCHITECTURES "native")

add_subdirectory(test)

install(TARGETS bf)
