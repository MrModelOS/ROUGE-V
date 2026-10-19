cmake_minimum_required(VERSION 3.20)
project(rouge-numerics LANGUAGES CXX)

# Волна 2 роадмапа: своя открытая математика (FP32 CPU GEMM + автотюнер)
# вместо зависимости от cuBLAS. Библиотека самостоятельная: её можно собрать
# и протестировать отдельно, без остального дерева ROUGE-V.
#
#   cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -S software/rouge-numerics
#   cmake --build build
#   ctest --test-dir build --output-on-failure

option(ROUGE_BUILD_TESTS "Build unit tests" ON)
# When pulled in as a subproject (rouge-cuda needs the cuBLAS compat shim),
# the parent can suppress the numerics unit tests with a normal variable.
if(DEFINED ROUGE_NUMERICS_NO_TESTS)
  set(ROUGE_BUILD_TESTS OFF)
endif()
# Опционально: собрать под инструкции этой машины (-march=native). По умолчанию
# выключено — иначе бинарь перестаёт работать на других процессорах. Без этого
# флага ядро само выбирает набор инструкций в рантайме (baseline/avx2/avx512).
option(ROUGE_NUMERICS_NATIVE "Build with -march=native" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(rouge-numerics STATIC
  src/gemm.cpp
  src/autotune.cpp
  src/fp8.cpp
  src/fp16.cpp
  src/narrow.cpp
  src/quant_int8.cpp
  src/attention_gqa.cpp
  src/transformer.cpp
  src/conv.cpp
  src/conv_implicit.cpp
  src/conv_backward.cpp
  src/winograd.cpp
  src/attention.cpp
  src/fft.cpp
  src/sparse_random.cpp
  src/complex_gemm.cpp
  src/blas1.cpp
)
target_include_directories(rouge-numerics PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
set_target_properties(rouge-numerics PROPERTIES POSITION_INDEPENDENT_CODE ON)

# split_k поднимает по одному std::thread на часть K — нужен pthread.
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(rouge-numerics PUBLIC Threads::Threads)

# Ядро считает в float32 и использует FMA: разрешаем контракцию a*b+c, иначе
# результат может отличаться от замеров в зависимости от флагов компилятора.
include(CheckCXXCompilerFlag)
foreach(flag -ffp-contract=fast -march=native)
  string(MAKE_C_IDENTIFIER "HAVE${flag}" flag_var)
  check_cxx_compiler_flag(${flag} ${flag_var})
endforeach()
if(HAVE-ffp-contract=fast)
  target_compile_options(rouge-numerics PRIVATE -ffp-contract=fast)
endif()
if(ROUGE_NUMERICS_NATIVE AND HAVE-march=native)
  target_compile_options(rouge-numerics PRIVATE -march=native)
endif()

target_compile_options(rouge-numerics PRIVATE -Wall -Wextra)

# Шимы чужих API поверх своего кода (Wave 2, подход Wine: интерфейсы по
# открытой документации, реализация своя). Отдельные shared-библиотеки, чтобы
# их можно было подсунуть линковщику вместо проприетарных.
add_library(rouge-cublas SHARED src/cublas_shim.cpp src/cublas_blas1.cpp)
target_include_directories(rouge-cublas PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(rouge-cublas PUBLIC rouge-numerics)
target_compile_options(rouge-cublas PRIVATE -Wall -Wextra)

add_library(rouge-cudnn SHARED src/cudnn_shim.cpp src/cudnn_conv_bwd.cpp)
target_include_directories(rouge-cudnn PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(rouge-cudnn PUBLIC rouge-numerics)
target_compile_options(rouge-cudnn PRIVATE -Wall -Wextra)

if(ROUGE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()