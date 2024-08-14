if(CCGLIB_BENCHMARK_WITH_PMT)
  set(PMT_BUILD_NVML ON)
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
  add_executable(benchmark_${benchmark} benchmark_${benchmark}.cu)

  target_link_libraries(benchmark_${benchmark} ccglib cudawrappers::cu
                        cudawrappers::nvrtc cxxopts)
  if(CCGLIB_BENCHMARK_WITH_PMT)
    target_link_libraries(benchmark_${benchmark} pmt)
  endif()

  target_include_directories(benchmark_${benchmark}
                             PRIVATE "${PROJECT_SOURCE_DIR}/include")
endforeach()
