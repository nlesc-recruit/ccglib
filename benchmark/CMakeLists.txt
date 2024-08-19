if(CCGLIB_BENCHMARK_WITH_PMT)
  if(${CCGLIB_BACKEND_HIP})
    set(PMT_BUILD_ROCM ON)
  else()
    set(PMT_BUILD_NVML ON)
  endif()
  FetchContent_Declare(pmt GIT_REPOSITORY https://git.astron.nl/RD/pmt)
  FetchContent_MakeAvailable(pmt)
  add_compile_definitions("HAVE_PMT")
endif()

FetchContent_Declare(
  cxxopts
  GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
  GIT_TAG v3.2.0)
FetchContent_MakeAvailable(cxxopts)

set(BENCHMARKS gemm packing transpose)
foreach(benchmark ${BENCHMARKS})
  set(TARGET benchmark_${benchmark})
  set(SOURCE benchmark_${benchmark}.cu)
  add_executable(${TARGET} ${SOURCE})

  target_link_libraries(${TARGET} ccglib cudawrappers::cu cudawrappers::nvrtc
                        cxxopts)
  if(CCGLIB_BENCHMARK_WITH_PMT)
    target_link_libraries(${TARGET} pmt)
  endif()

  target_include_directories(${TARGET} PRIVATE "${PROJECT_SOURCE_DIR}/include")

  if(${CCGLIB_BACKEND_HIP})
    set_source_files_properties(${SOURCE} PROPERTIES LANGUAGE HIP)
  endif()

endforeach()
