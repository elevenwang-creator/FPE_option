cmake_minimum_required(VERSION 3.16)
project(fpe_examples LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT FPE_PREFIX)
  message(FATAL_ERROR
    "FPE_PREFIX not set. Pass -DFPE_PREFIX=<pixi env prefix>\n"
    " e.g.: cmake -DFPE_PREFIX=$(pixi env prefix) .."
  )
endif()

set(FPE_LIB_DIR "${FPE_PREFIX}/lib")
set(FPE_INC_DIR "${FPE_PREFIX}/include")

add_executable(demo demo.cpp)
target_include_directories(demo PRIVATE ${FPE_INC_DIR})
target_link_directories(demo PRIVATE ${FPE_LIB_DIR})
target_link_libraries(demo PRIVATE fpe_engine)
set_target_properties(demo PROPERTIES
  BUILD_RPATH "${FPE_LIB_DIR}"
  INSTALL_RPATH "${FPE_LIB_DIR}"
)
