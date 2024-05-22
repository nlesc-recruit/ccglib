project(helper)

add_library(${PROJECT_NAME} OBJECT Transpose.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE ${CCGLIB_INCLUDE_DIR})

target_link_libraries(${PROJECT_NAME} PRIVATE cudawrappers::cu
                                              cudawrappers::nvrtc)
