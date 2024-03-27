project(gemm-reference)

add_library(${PROJECT_NAME} OBJECT GEMM.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart OpenMP::OpenMP_CXX
                                              xtensor)
