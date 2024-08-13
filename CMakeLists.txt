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

include(GNUInstallDirs)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  # Install project cmake targets
  include(CMakePackageConfigHelpers)

  # Install library and public header
  install(
    TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}
    PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}")

  # Install cmake config files
  install(
    EXPORT ${PROJECT_NAME}
    FILE ${PROJECT_NAME}-exported.cmake
    EXPORT_LINK_INTERFACE_LIBRARIES
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})

  configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
    NO_SET_AND_CHECK_MACRO NO_CHECK_REQUIRED_COMPONENTS_MACRO)

  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake"
          DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")
endif()
