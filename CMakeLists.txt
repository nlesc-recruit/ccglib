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
  GIT_TAG 0.6.0)
FetchContent_MakeAvailable(cudawrappers)

include(cmake/xtensor.cmake)

include(cmake/helper.cmake)

set(INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_PREFIX}/include")
configure_file(cmake/config.h.in config.h)

# these files are needed during runtime compilation
set(RUNTIME_INCLUDE_FILES ${CMAKE_SOURCE_DIR}/include/ccglib/wmma_extension.h
                          ${CMAKE_SOURCE_DIR}/include/ccglib/async_copies.h)
create_symlinks(${CMAKE_CURRENT_BINARY_DIR} "${RUNTIME_INCLUDE_FILES}")

# build beamformer
add_executable(bf src/bf.cu)
target_include_directories(bf PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
                                      ${CMAKE_SOURCE_DIR}/kernels)
target_link_libraries(bf PRIVATE cudawrappers::cu cudawrappers::nvrtc
                                 OpenMP::OpenMP_CXX xtensor)
target_embed_source(bf kernels/gemm_kernel.cu)
set_target_properties(bf PROPERTIES CUDA_ARCHITECTURES "native")

install(TARGETS bf)
install(FILES "${RUNTIME_INCLUDE_FILES}" DESTINATION include)
