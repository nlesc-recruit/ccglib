project(gemm-reference)

add_library(${PROJECT_NAME} OBJECT GEMM.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE ${CCGLIB_INCLUDE_DIR})

target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart OpenMP::OpenMP_CXX
                                              xtensor cudawrappers::cu)
