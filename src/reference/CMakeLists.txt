project(gemm-reference)

add_library(${PROJECT_NAME} OBJECT GEMM.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE ${CCGLIB_INCLUDE_DIR})

target_link_libraries(
  ${PROJECT_NAME} PRIVATE LINK_gpu_runtime OpenMP::OpenMP_CXX xtensor
                          cudawrappers::cu)

if(${CCGLIB_BACKEND_HIP})
  get_target_property(sources ${PROJECT_NAME} SOURCES)
  set_source_files_properties(${sources} PROPERTIES LANGUAGE HIP)
endif()
