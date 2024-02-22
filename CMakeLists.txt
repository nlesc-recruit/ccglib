cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(
    complex_wmma_gemmm
    DESCRIPTION "COMPLEX WMMA GEMM"
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

# build beamformer
add_executable(bf bf.cu)
target_link_libraries(bf PRIVATE cudawrappers::cu cudawrappers::nvrtc OpenMP::OpenMP_CXX)
target_embed_source(bf gemm_kernel.cu)
# cmake doesn't want to add -fopenmp for some reason, so do it manually
target_compile_options(bf PRIVATE -Wall -fopenmp)

install(TARGETS bf)
install(FILES wmma_extension.h async_copies.h DESTINATION include)
