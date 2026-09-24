cmake_minimum_required(VERSION 3.20)
project(rouge-cuda LANGUAGES CXX)

option(ROUGE_BUILD_TESTS "Build unit tests" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra)

# ---- Dependencies ----
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../rouge-ptx ${CMAKE_CURRENT_BINARY_DIR}/rouge-ptx)

# ---- Shared library: interposes CUDA symbols (LD_PRELOAD-ready) ----
add_library(rouge-cuda SHARED
  src/driver_api.cpp
  src/runtime_api.cpp
  src/forward.cpp
)
target_include_directories(rouge-cuda PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(rouge-cuda PRIVATE rouge-ptx ${CMAKE_DL_LIBS})
set_target_properties(rouge-cuda PROPERTIES OUTPUT_NAME rouge-cuda)

# ---- Static library: for embedding in our own software stack and tests ----
add_library(rouge-cuda-static STATIC
  src/driver_api.cpp
  src/runtime_api.cpp
  src/forward.cpp
)
target_include_directories(rouge-cuda-static PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(rouge-cuda-static PUBLIC rouge-ptx ${CMAKE_DL_LIBS})
set_target_properties(rouge-cuda-static PROPERTIES OUTPUT_NAME rouge-cuda-static)

if(ROUGE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()