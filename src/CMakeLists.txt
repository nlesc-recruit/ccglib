# Set path to local include directory
get_filename_component(CCGLIB_INCLUDE_DIR
                       "${CMAKE_CURRENT_LIST_DIR}/../include" ABSOLUTE)

# Build the library components
add_subdirectory(reference)
add_subdirectory(mma)
add_subdirectory(transpose)
add_subdirectory(packing)

# Link the main library
add_library(
  ${PROJECT_NAME} SHARED
  $<TARGET_OBJECTS:gemm-reference> $<TARGET_OBJECTS:transpose>
  $<TARGET_OBJECTS:gemm-mma> $<TARGET_OBJECTS:packing>)

target_link_libraries(ccglib PUBLIC cudawrappers::cu cudawrappers::nvrtc)
target_link_libraries(ccglib PRIVATE OpenMP::OpenMP_CXX xtensor)

target_embed_source(ccglib ../kernels/gemm_kernel_float.cu)
target_embed_source(ccglib ../kernels/gemm_kernel_int1.cu)
target_embed_source(ccglib ../kernels/transpose_kernel.cu)
target_embed_source(ccglib ../kernels/packing_kernel.cu)

target_include_directories(ccglib INTERFACE ${CCGLIB_INCLUDE_DIR})
