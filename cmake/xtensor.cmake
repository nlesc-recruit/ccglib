FetchContent_Declare(
  xtl
  GIT_REPOSITORY https://github.com/xtensor-stack/xtl.git
  GIT_TAG 0.7.7
  GIT_SHALLOW TRUE)
FetchContent_Populate(xtl)

FetchContent_Declare(
  xtensor
  GIT_REPOSITORY https://github.com/xtensor-stack/xtensor.git
  GIT_TAG 0.25.0
  GIT_SHALLOW TRUE)
FetchContent_Populate(xtensor)

add_library(xtl INTERFACE)
target_include_directories(xtl SYSTEM
                           INTERFACE "${FETCHCONTENT_BASE_DIR}/xtl-src/include")

add_library(xtensor INTERFACE)
target_include_directories(
  xtensor SYSTEM INTERFACE "${FETCHCONTENT_BASE_DIR}/xtensor-src/include")
target_link_libraries(xtensor INTERFACE xtl)
