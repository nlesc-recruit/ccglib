FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.5.4)

FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/contrib)
include(Catch)

add_executable(test-helper helper.cpp)
target_include_directories(test-helper PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_include_directories(test-helper PRIVATE ${CMAKE_BINARY_DIR})
target_link_libraries(test-helper PRIVATE cudawrappers::cu Catch2WithMain)

add_executable(test-reference gemm/reference.cpp)
target_include_directories(test-reference
                           PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_include_directories(test-reference PRIVATE ${CMAKE_BINARY_DIR})
target_link_libraries(test-reference PRIVATE ccglib cudawrappers::cu
                                             Catch2WithMain xtensor)

add_executable(test-mma gemm/mma.cpp)
target_include_directories(test-mma PRIVATE ${CMAKE_BINARY_DIR}
                                            "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(test-mma PRIVATE ccglib cudawrappers::cu
                                       Catch2::Catch2WithMain xtensor)

add_executable(test-packing packing/packing.cpp)
target_include_directories(test-packing PRIVATE ${CMAKE_BINARY_DIR}
                                                "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(test-packing PRIVATE ccglib cudawrappers::cu
                                           Catch2::Catch2WithMain)

add_executable(test-transpose transpose/transpose.cpp)
target_include_directories(
  test-transpose PRIVATE ${CMAKE_BINARY_DIR} "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(test-transpose PRIVATE ccglib cudawrappers::cu
                                             Catch2::Catch2WithMain xtensor)

set(TESTS test-helper test-reference test-mma test-packing test-transpose)

foreach(test IN ITEMS ${TESTS})
  add_test(NAME ${test} COMMAND ${test})
  if(${CCGLIB_BACKEND_HIP})
    get_target_property(sources ${test} SOURCES)
    set_source_files_properties(${sources} PROPERTIES LANGUAGE HIP)
  endif()
endforeach()
