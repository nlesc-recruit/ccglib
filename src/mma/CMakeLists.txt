project(gemm-mma)

add_library(${PROJECT_NAME} OBJECT GEMM.cpp Kernel.cpp KernelFloat16.cpp
                                   KernelInt1.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE ${CCGLIB_INCLUDE_DIR})

target_link_libraries(${PROJECT_NAME} PRIVATE cudawrappers::cu
                                              cudawrappers::nvrtc)
