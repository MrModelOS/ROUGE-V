cmake_minimum_required(VERSION 3.20)
project(ROUGE-V LANGUAGES CXX)

enable_testing()

# Root build aggregates the AOT compiler pipeline (rouge-compiler) and the
# numerics library (rouge-numerics). For component-only builds use
# -S software/rouge-compiler / -S software/rouge-cuda / -S software/rouge-numerics.
add_subdirectory(software/rouge-compiler ${CMAKE_BINARY_DIR}/rouge-compiler)
add_subdirectory(software/rouge-numerics ${CMAKE_BINARY_DIR}/rouge-numerics)
