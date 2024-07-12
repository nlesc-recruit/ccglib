# List of headers to install. Append to it below.
set(CCGLIB_HEADERS "")

# Set path to local include directory
get_filename_component(CCGLIB_INCLUDE_DIR
                       "${CMAKE_CURRENT_LIST_DIR}/../include" ABSOLUTE)

# Build the library components
add_subdirectory(reference)
list(APPEND CCGLIB_HEADERS "gemm/reference.h")

add_subdirectory(mma)
list(APPEND CCGLIB_HEADERS "gemm/mma.h")

add_subdirectory(transpose)
list(APPEND CCGLIB_HEADERS "transpose/transpose.h")

add_subdirectory(packing)
list(APPEND CCGLIB_HEADERS "packing/packing.h")

# Link the main library
add_library(
  ${PROJECT_NAME} SHARED
  $<TARGET_OBJECTS:gemm-reference> $<TARGET_OBJECTS:transpose>
  $<TARGET_OBJECTS:gemm-mma> $<TARGET_OBJECTS:packing>)

target_link_libraries(ccglib PRIVATE cudawrappers::cu cudawrappers::nvrtc
                                     OpenMP::OpenMP_CXX xtensor)

target_embed_source(ccglib ../kernels/gemm_kernel_float.cu)
target_embed_source(ccglib ../kernels/gemm_kernel_int1.cu)
target_embed_source(ccglib ../kernels/transpose_kernel.cu)
target_embed_source(ccglib ../kernels/packing_kernel.cu)

# You can use ccglib by installing it, or by including it in your project
set(CCGLIB_INSTALLED ${CMAKE_PROJECT_NAME} STREQUAL ${PROJECT_NAME})

if(${CCGLIB_INSTALLED})
  # Set target include directory
  target_include_directories(
    ccglib INTERFACE $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
  set(CCGLIB_PUBLIC_HEADER "${CMAKE_SOURCE_DIR}/include/ccglib/ccglib.hpp")

  # Set target public header (to have it installed automatically)
  set_target_properties(ccglib PROPERTIES PUBLIC_HEADER ${CCGLIB_PUBLIC_HEADER})

  # Install all the other public header files manually
  set(CCGLIB_HEADER_PREFIX "${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}")
  foreach(ccglib_HEADER IN LISTS CCGLIB_HEADERS)
    get_filename_component(ccglib_HEADER_SUBDIRECTORY "${ccglib_HEADER}"
                           DIRECTORY)
    install(FILES "${CMAKE_SOURCE_DIR}/include/ccglib/${ccglib_HEADER}"
            DESTINATION "${CCGLIB_HEADER_PREFIX}/${ccglib_HEADER_SUBDIRECTORY}")
  endforeach()
else()
  # When the library is included, the local include directory is exposed
  target_include_directories(ccglib INTERFACE ${CCGLIB_INCLUDE_DIR})
endif()
