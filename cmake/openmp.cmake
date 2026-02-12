find_package(OpenMP REQUIRED)

# Workaround for broken OpenMP::OpenMP_CXX target with Clang
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  if(TARGET OpenMP::OpenMP_CXX)
    # Clear target properties for broken target, if it exists
    set_target_properties(
      OpenMP::OpenMP_CXX
      PROPERTIES INTERFACE_COMPILE_OPTIONS ""
                 INTERFACE_LINK_LIBRARIES ""
                 INTERFACE_INCLUDE_DIRECTORIES "")
  else()
    # Create the target if it doesn't exist
    add_library(OpenMP::OpenMP_CXX INTERFACE IMPORTED)
  endif()
endif()
