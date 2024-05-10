FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.5.4)

FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/contrib)
include(Catch)

add_executable(test-helper helper.cpp)
target_include_directories(test-helper PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_include_directories(test-helper PRIVATE ${CMAKE_BINARY_DIR})
target_link_libraries(test-helper PRIVATE cudawrappers::cu Catch2WithMain)

add_executable(test-reference reference.cpp)
target_include_directories(test-reference PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_include_directories(test-reference PRIVATE ${CMAKE_BINARY_DIR})
target_link_libraries(
  test-reference PRIVATE cudawrappers::cu xtensor OpenMP::OpenMP_CXX
                         $<TARGET_OBJECTS:gemm-reference> Catch2WithMain)

add_executable(test-gemm gemm.cpp)
target_include_directories(
  test-gemm PRIVATE ${CMAKE_BINARY_DIR} "${CMAKE_SOURCE_DIR}/src"
                    "${CMAKE_SOURCE_DIR}/kernels")
target_link_libraries(
  test-gemm
  PRIVATE cudawrappers::cu
          cudawrappers::nvrtc
          OpenMP::OpenMP_CXX
          xtensor
          $<TARGET_OBJECTS:gemm-reference>
          $<TARGET_OBJECTS:helper>
          $<TARGET_OBJECTS:gemm-mma>
          Catch2::Catch2WithMain)
target_embed_source(test-gemm ../kernels/gemm_kernel.cu)
target_embed_source(test-gemm ../kernels/transpose_kernel.cu)

set(TESTS test-helper test-reference test-gemm)

foreach(test IN ITEMS ${TESTS})
  add_test(NAME ${test} COMMAND ${test})
endforeach()
