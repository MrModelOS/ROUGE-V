cmake_minimum_required(VERSION 3.20)
project(ROUGE-V LANGUAGES CXX)

enable_testing()

# Root build aggregates the AOT compiler pipeline (rouge-compiler), the
# numerics library (rouge-numerics) and the renderer (rouge-render).
# For component-only builds use
# -S software/rouge-compiler / -S software/rouge-cuda / -S software/rouge-numerics
# -S software/rouge-render.
add_subdirectory(software/rouge-compiler ${CMAKE_BINARY_DIR}/rouge-compiler)
add_subdirectory(software/rouge-numerics ${CMAKE_BINARY_DIR}/rouge-numerics)
add_subdirectory(software/rouge-render ${CMAKE_BINARY_DIR}/rouge-render)
