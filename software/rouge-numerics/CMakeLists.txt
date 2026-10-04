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

if(ROUGE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()