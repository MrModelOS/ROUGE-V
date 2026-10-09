add_executable(test_gemm test_gemm.cpp)
target_link_libraries(test_gemm PRIVATE rouge-numerics)
target_compile_options(test_gemm PRIVATE -Wall -Wextra)
add_test(NAME gemm COMMAND test_gemm)

# Тот же набор конфигов, но для КАЖДОГО собранного варианта ISA: на одной
# машине процессор умеет не всё, и без этого проверялся бы ровно один путь.
add_executable(test_gemm_isa test_gemm_isa.cpp)
target_link_libraries(test_gemm_isa PRIVATE rouge-numerics)
target_compile_options(test_gemm_isa PRIVATE -Wall -Wextra)
foreach(isa base avx2 avx512)
  add_test(NAME gemm_isa_${isa} COMMAND test_gemm_isa)
  set_tests_properties(gemm_isa_${isa} PROPERTIES
    ENVIRONMENT "ROUGE_NUMERICS_ISA=${isa}")
endforeach()

add_executable(test_gemm_fp8 test_gemm_fp8.cpp)
target_link_libraries(test_gemm_fp8 PRIVATE rouge-numerics)
target_compile_options(test_gemm_fp8 PRIVATE -Wall -Wextra)
add_test(NAME gemm_fp8 COMMAND test_gemm_fp8)

add_executable(test_conv test_conv.cpp)
target_link_libraries(test_conv PRIVATE rouge-numerics)
target_compile_options(test_conv PRIVATE -Wall -Wextra)
add_test(NAME conv COMMAND test_conv)
