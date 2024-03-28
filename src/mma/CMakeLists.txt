project(gemm-mma)

add_library(${PROJECT_NAME} OBJECT GEMM.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_BINARY_DIR})
target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_SOURCE_DIR}/src")

target_link_libraries(${PROJECT_NAME} PRIVATE cudawrappers::cu
                                              cudawrappers::nvrtc)
