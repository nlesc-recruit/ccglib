cmake_minimum_required(VERSION 3.20)

project(example)

find_package(CUDAToolkit REQUIRED)

include(FetchContent)

FetchContent_Declare(
  cudawrappers
  GIT_REPOSITORY https://github.com/nlesc-recruit/cudawrappers
  GIT_TAG main)
FetchContent_MakeAvailable(cudawrappers)

if(NOT DEFINED CCGLIB_GIT_REVISION)
  set(CCGLIB_GIT_REVISION "origin/main")
endif()

FetchContent_Declare(
  ccglib
  GIT_REPOSITORY https://git.astron.nl/RD/recruit/ccglib
  GIT_TAG ${CCGLIB_GIT_REVISION})
FetchContent_MakeAvailable(ccglib)

add_executable(${PROJECT_NAME} example.cpp)
target_link_libraries(${PROJECT_NAME} ccglib cudawrappers::cu
                      cudawrappers::nvrtc)
