cmake_minimum_required(VERSION 3.20)
project(ROUGE-V LANGUAGES CXX)

enable_testing()

# Root build aggregates the AOT compiler pipeline (rouge-compiler).
# For component-only builds use -S software/rouge-compiler or -S software/rouge-cuda.
add_subdirectory(software/rouge-compiler ${CMAKE_BINARY_DIR}/rouge-compiler)
