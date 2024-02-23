FetchContent_Declare(
  xtl
  GIT_REPOSITORY https://github.com/xtensor-stack/xtl.git
  GIT_TAG 0.7.7
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(xtl)

FetchContent_Declare(
  xsimd
  GIT_REPOSITORY https://github.com/xtensor-stack/xsimd.git
  GIT_TAG 12.1.1
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(xsimd)

FetchContent_Declare(
  xtensor
  GIT_REPOSITORY https://github.com/xtensor-stack/xtensor.git
  GIT_TAG 0.25.0
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(xtensor)
