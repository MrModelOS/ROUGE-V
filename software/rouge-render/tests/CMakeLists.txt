add_executable(test_render test_render.cpp)
target_link_libraries(test_render PRIVATE rouge-render)
target_compile_options(test_render PRIVATE -Wall -Wextra)
add_test(NAME render COMMAND test_render)
