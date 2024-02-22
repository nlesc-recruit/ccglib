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

# fetch cudawrappers
include(FetchContent)
FetchContent_Declare(
  cudawrappers
  GIT_REPOSITORY https://github.com/nlesc-recruit/cudawrappers
  GIT_TAG 0.6.0)
FetchContent_MakeAvailable(cudawrappers)

set(INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_PREFIX}/include")
configure_file(cmake/config.h.in config.h)

# build beamformer
add_executable(bf bf.cu)
target_include_directories(bf PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(bf PRIVATE cudawrappers::cu cudawrappers::nvrtc
                                 OpenMP::OpenMP_CXX)
target_embed_source(bf gemm_kernel.cu)
set_target_properties(bf PROPERTIES CUDA_ARCHITECTURES "native")

install(TARGETS bf)
install(FILES wmma_extension.h async_copies.h DESTINATION include)
