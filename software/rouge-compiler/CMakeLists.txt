cmake_minimum_required(VERSION 3.20)
project(rouge-compiler LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra)

# Shared PTX front-end (parser), also used by the rouge-cuda compat layer.
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../rouge-ptx ${CMAKE_CURRENT_BINARY_DIR}/rouge-ptx)

# ---- Translator library (PTX -> LLVM IR) ----
add_library(rouge-compiler-lib STATIC src/ptx_to_llvm.cpp)
target_include_directories(rouge-compiler-lib PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(rouge-compiler-lib PUBLIC rouge-ptx)

# ---- CLI: ptx2ir <kernel.ptx> [out.ll] ----
add_executable(ptx2ir src/ptx2ir_main.cpp)
target_link_libraries(ptx2ir PRIVATE rouge-compiler-lib)

# ---- CLI: rouge-run <kernel.ptx> [-o kernel.o] ----
# Translates a kernel and verifies the result survives a real backend.
add_executable(rouge-run src/rouge_run_main.cpp)
target_link_libraries(rouge-run PRIVATE rouge-compiler-lib)

# ---- Host runtime: C-API helpers for AOT kernels (barriers, launch state) ----
# Seed of the libcudart/libcuda replacement; host pthread backend.
find_package(Threads REQUIRED)
add_library(rouge-runtime STATIC runtime/rouge_runtime.cpp)
target_include_directories(rouge-runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/runtime)
target_link_libraries(rouge-runtime PUBLIC Threads::Threads)

enable_testing()

# ---- MLIR contour (stage 2: GPU -> Vector -> RVV), optional ----
# Requires an MLIR package; the PTX->LLVM IR path above does not.
option(ROUGE_ENABLE_MLIR "Build the MLIR contour passes (needs an MLIR package)" OFF)
if(ROUGE_ENABLE_MLIR)
  add_subdirectory(mlir)
endif()

add_subdirectory(tests)