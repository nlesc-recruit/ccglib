cmake_minimum_required(VERSION 3.20)

project(example)

option(USE_HIP "Use HIP backend instead of CUDA" False)

if(${USE_HIP})
  enable_language(HIP)
  set(CUDAWRAPPERS_BACKEND "HIP")
  set(CCGLIB_BACKEND "HIP")
else()
  find_package(CUDAToolkit REQUIRED)
  enable_language(CUDA)
endif()

include(FetchContent)

FetchContent_Declare(
  cudawrappers
  GIT_REPOSITORY https://github.com/nlesc-recruit/cudawrappers
  GIT_TAG "main")
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

if(${USE_HIP})
  # hip-specific example
  add_executable(${PROJECT_NAME}_hip example.hip)
  target_link_libraries(${PROJECT_NAME}_hip ccglib)

  # ensure generic example is compiled with hip
  set_source_files_properties(example.cpp PROPERTIES LANGUAGE HIP)
else()
  # cuda-specific example
  add_executable(${PROJECT_NAME}_cuda example.cu)
  target_link_libraries(${PROJECT_NAME}_cuda ccglib)
endif()
