cmake_minimum_required(VERSION 3.20)
project(rouge-ptx LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra)

# Shared PTX front-end: parser + interpreter (used by rouge-cuda compat layer
# and the rouge-compiler AOT pipeline).
add_library(rouge-ptx STATIC
  ptx.cpp
)
set_target_properties(rouge-ptx PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(rouge-ptx PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})