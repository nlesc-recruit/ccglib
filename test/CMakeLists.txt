FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.5.4
)

FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/contrib)
include(Catch)

add_executable(test-helper
  helper.cpp
)
target_include_directories(test-helper PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_include_directories(test-helper PRIVATE ${CMAKE_BINARY_DIR})
target_link_libraries(test-helper PRIVATE cudawrappers::cu
                                          Catch2WithMain)

add_executable(test-reference
  reference.cpp
)
target_include_directories(test-reference PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_include_directories(test-reference PRIVATE ${CMAKE_BINARY_DIR})
target_link_libraries(test-reference PRIVATE cudawrappers::cu
                                             xtensor
                                             OpenMP::OpenMP_CXX
                                             $<TARGET_OBJECTS:gemm-reference>
                                             Catch2WithMain)

set(TESTS test-helper test-reference)

foreach(TEST IN ITEMS ${TESTS})
  add_test(${TEST} ${TEST})
endforeach()