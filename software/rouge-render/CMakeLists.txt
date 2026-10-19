cmake_minimum_required(VERSION 3.20)
project(rouge-render LANGUAGES CXX)

# Волна 4 роадмапа: своя графика — path tracer, денойзер и темпоральный
# апскейлер вместо OptiX/DLSS. Никаких чужих библиотек: ни DLSS, ни OptiX,
# ни их нейросетей. Только свой код; измеряем качество сами (PSNR/SSIM).
#
#   cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -S software/rouge-render
#   cmake --build build
#   ctest --test-dir build --output-on-failure

option(ROUGE_BUILD_TESTS "Build unit tests" ON)
if(DEFINED ROUGE_NUMERICS_NO_TESTS)
  set(ROUGE_BUILD_TESTS OFF)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# RNG берём из numerics (тот же splitmix64, что во всём стеке) — чтобы все
# проекты давали одинаковую последовательность случайных чисел. Если
# библиотека уже собрана выше по дереву (корневой CMakeLists), используем её.
if(NOT TARGET rouge-numerics)
  set(ROUGE_NUMERICS_NO_TESTS ON)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../rouge-numerics
                   ${CMAKE_CURRENT_BINARY_DIR}/rouge-numerics)
endif()

add_library(rouge-render STATIC
  src/pathtrace.cpp
  src/denoise.cpp
  src/upsample.cpp
  src/metrics.cpp
)
target_include_directories(rouge-render PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(rouge-render PUBLIC rouge-numerics)
target_compile_options(rouge-render PRIVATE -Wall -Wextra)
set_target_properties(rouge-render PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(ROUGE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
