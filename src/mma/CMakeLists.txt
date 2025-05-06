project(gemm-mma)

add_library(
  ${PROJECT_NAME} OBJECT GEMM.cpp Kernel.cpp KernelFloat16.cpp
                         KernelBfloat16.cpp KernelFloat32.cpp KernelInt1.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE ${CCGLIB_INCLUDE_DIR})

target_link_libraries(${PROJECT_NAME} PRIVATE cudawrappers::cu
                                              cudawrappers::nvrtc)

if(${CCGLIB_BACKEND_HIP})
  get_target_property(sources ${PROJECT_NAME} SOURCES)
  set_source_files_properties(${sources} PROPERTIES LANGUAGE HIP)
endif()
