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

add_executable(test_gemm_f16 test_gemm_f16.cpp)
target_link_libraries(test_gemm_f16 PRIVATE rouge-numerics)
target_compile_options(test_gemm_f16 PRIVATE -Wall -Wextra)
add_test(NAME gemm_f16 COMMAND test_gemm_f16)

add_executable(test_conv test_conv.cpp)
target_link_libraries(test_conv PRIVATE rouge-numerics)
target_compile_options(test_conv PRIVATE -Wall -Wextra)
add_test(NAME conv COMMAND test_conv)

add_executable(test_winograd test_winograd.cpp)
target_link_libraries(test_winograd PRIVATE rouge-numerics)
target_compile_options(test_winograd PRIVATE -Wall -Wextra)
add_test(NAME winograd COMMAND test_winograd)

add_executable(test_attention test_attention.cpp)
target_link_libraries(test_attention PRIVATE rouge-numerics)
target_compile_options(test_attention PRIVATE -Wall -Wextra)
add_test(NAME attention COMMAND test_attention)

add_executable(test_fft_sparse_rng test_fft_sparse_rng.cpp)
target_link_libraries(test_fft_sparse_rng PRIVATE rouge-numerics)
target_compile_options(test_fft_sparse_rng PRIVATE -Wall -Wextra)
add_test(NAME fft_sparse_rng COMMAND test_fft_sparse_rng)

add_executable(test_shims test_shims.cpp)
target_link_libraries(test_shims PRIVATE rouge-cublas rouge-cudnn rouge-numerics)
target_compile_options(test_shims PRIVATE -Wall -Wextra)
add_test(NAME shims COMMAND test_shims)

# Замер против системного BLAS (если есть cblas.h + библиотека — меряем
# cblas_sgemm, иначе только наивный пол). Скорость — отчёт, падает тест
# только на неверном результате.
find_path(CBLAS_INCLUDE_DIR cblas.h)
find_library(CBLAS_LIBRARY NAMES cblas blas)
if(CBLAS_INCLUDE_DIR AND CBLAS_LIBRARY)
  add_executable(bench_gemm_blas bench_gemm_blas.cpp)
  target_link_libraries(bench_gemm_blas PRIVATE rouge-numerics ${CBLAS_LIBRARY})
  target_include_directories(bench_gemm_blas PRIVATE ${CBLAS_INCLUDE_DIR})
  target_compile_definitions(bench_gemm_blas PRIVATE ROUGE_HAVE_CBLAS=1)
  target_compile_options(bench_gemm_blas PRIVATE -Wall -Wextra)
  add_test(NAME bench_blas COMMAND bench_gemm_blas)
else()
  add_executable(bench_gemm_blas bench_gemm_blas.cpp)
  target_link_libraries(bench_gemm_blas PRIVATE rouge-numerics)
  target_compile_options(bench_gemm_blas PRIVATE -Wall -Wextra)
  add_test(NAME bench_blas COMMAND bench_gemm_blas)
endif()
