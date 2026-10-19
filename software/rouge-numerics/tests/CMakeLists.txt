add_executable(test_gemm test_gemm.cpp)
target_link_libraries(test_gemm PRIVATE rouge-numerics)
target_compile_options(test_gemm PRIVATE -Wall -Wextra)
add_test(NAME gemm COMMAND test_gemm)

# Внешние эталоны — только для тестов (библиотека от них не зависит):
#   * BLAS: OpenBLAS предпочтительно (ROUGE_OPENBLAS_ROOT — локальный префикс
#     с openblas/{lib,include/openblas}), иначе системный CBLAS (это обычно
#     reference BLAS — мы это печатаем, чтобы не выдавать его за OpenBLAS);
#   * oneDNN (ROUGE_ONEDNN_ROOT) — эталон для свёртки;
#   * FFTW single — эталон для FFT.
set(ROUGE_OPENBLAS_ROOT "" CACHE PATH
    "Local prefix with OpenBLAS: <prefix>/lib/libopenblas.so, <prefix>/include/openblas/cblas.h")
set(ROUGE_ONEDNN_ROOT "" CACHE PATH
    "Local prefix with oneDNN: <prefix>/lib/libdnnl.so, <prefix>/include/oneapi/dnnl/dnnl.h")
if(ROUGE_OPENBLAS_ROOT)
  find_library(OPENBLAS_LIBRARY NAMES openblas
    HINTS ${ROUGE_OPENBLAS_ROOT}/lib NO_DEFAULT_PATH)
  find_path(OPENBLAS_INCLUDE_DIR cblas.h
    HINTS ${ROUGE_OPENBLAS_ROOT}/include/openblas NO_DEFAULT_PATH)
endif()
if(OPENBLAS_LIBRARY AND OPENBLAS_INCLUDE_DIR)
  set(BLAS_LIB ${OPENBLAS_LIBRARY})
  set(BLAS_INC ${OPENBLAS_INCLUDE_DIR})
  set(BLAS_NAME "openblas")
else()
  find_path(CBLAS_INCLUDE_DIR cblas.h)
  find_library(CBLAS_LIBRARY NAMES cblas blas)
  if(CBLAS_LIBRARY AND CBLAS_INCLUDE_DIR)
    set(BLAS_LIB ${CBLAS_LIBRARY})
    set(BLAS_INC ${CBLAS_INCLUDE_DIR})
    set(BLAS_NAME "system-cblas")
  endif()
endif()
if(ROUGE_ONEDNN_ROOT)
  find_library(DNNL_LIBRARY NAMES dnnl
    HINTS ${ROUGE_ONEDNN_ROOT}/lib NO_DEFAULT_PATH)
  find_path(DNNL_INCLUDE_DIR oneapi/dnnl/dnnl.h
    HINTS ${ROUGE_ONEDNN_ROOT}/include NO_DEFAULT_PATH)
endif()
find_path(FFTW_INCLUDE_DIR fftw3.h)
find_library(FFTWF_LIBRARY NAMES fftw3f)

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

add_executable(test_narrow test_narrow.cpp)
target_link_libraries(test_narrow PRIVATE rouge-numerics)
target_compile_options(test_narrow PRIVATE -Wall -Wextra)
add_test(NAME narrow COMMAND test_narrow)

add_executable(test_gemm_f16 test_gemm_f16.cpp)
target_link_libraries(test_gemm_f16 PRIVATE rouge-numerics)
target_compile_options(test_gemm_f16 PRIVATE -Wall -Wextra)
add_test(NAME gemm_f16 COMMAND test_gemm_f16)

add_executable(test_gemm_f64 test_gemm_f64.cpp)
target_link_libraries(test_gemm_f64 PRIVATE rouge-numerics)
target_compile_options(test_gemm_f64 PRIVATE -Wall -Wextra)
add_test(NAME gemm_f64 COMMAND test_gemm_f64)
# FP64 идёт тем же ISA-диспетчером — проверить все три пути, как для f32.
foreach(isa base avx2 avx512)
  add_test(NAME gemm_f64_isa_${isa} COMMAND test_gemm_f64)
  set_tests_properties(gemm_f64_isa_${isa} PROPERTIES
    ENVIRONMENT "ROUGE_NUMERICS_ISA=${isa}")
endforeach()

add_executable(test_complex test_complex.cpp)
target_link_libraries(test_complex PRIVATE rouge-numerics)
target_compile_options(test_complex PRIVATE -Wall -Wextra)
add_test(NAME complex COMMAND test_complex)

add_executable(test_conv test_conv.cpp)
target_link_libraries(test_conv PRIVATE rouge-numerics)
target_compile_options(test_conv PRIVATE -Wall -Wextra)
add_test(NAME conv COMMAND test_conv)

add_executable(test_winograd test_winograd.cpp)
target_link_libraries(test_winograd PRIVATE rouge-numerics)
target_compile_options(test_winograd PRIVATE -Wall -Wextra)
add_test(NAME winograd COMMAND test_winograd)

add_executable(test_implicit test_implicit.cpp)
target_link_libraries(test_implicit PRIVATE rouge-numerics)
target_compile_options(test_implicit PRIVATE -Wall -Wextra)
add_test(NAME implicit COMMAND test_implicit)

add_executable(test_attention test_attention.cpp)
target_link_libraries(test_attention PRIVATE rouge-numerics)
target_compile_options(test_attention PRIVATE -Wall -Wextra)
add_test(NAME attention COMMAND test_attention)

add_executable(test_fft_sparse_rng test_fft_sparse_rng.cpp)
target_link_libraries(test_fft_sparse_rng PRIVATE rouge-numerics)
target_compile_options(test_fft_sparse_rng PRIVATE -Wall -Wextra)
add_test(NAME fft_sparse_rng COMMAND test_fft_sparse_rng)

add_executable(test_ai test_ai.cpp)
target_link_libraries(test_ai PRIVATE rouge-numerics)
target_compile_options(test_ai PRIVATE -Wall -Wextra)
add_test(NAME ai COMMAND test_ai)

# Замеры AI-части: int8 GEMM, GQA-внимание, трансформерный блок.
add_executable(bench_ai bench_ai.cpp)
target_link_libraries(bench_ai PRIVATE rouge-numerics)
target_compile_options(bench_ai PRIVATE -Wall -Wextra)
add_test(NAME bench_ai COMMAND bench_ai)

add_executable(test_blas1 test_blas1.cpp)
target_link_libraries(test_blas1 PRIVATE rouge-numerics)
target_compile_options(test_blas1 PRIVATE -Wall -Wextra)
if(BLAS_LIB AND BLAS_INC)
  target_link_libraries(test_blas1 PRIVATE ${BLAS_LIB})
  target_include_directories(test_blas1 PRIVATE ${BLAS_INC})
  target_compile_definitions(test_blas1 PRIVATE ROUGE_HAVE_CBLAS=1)
endif()
add_test(NAME blas1 COMMAND test_blas1)

add_executable(test_bwd test_bwd.cpp)
target_link_libraries(test_bwd PRIVATE rouge-numerics)
target_compile_options(test_bwd PRIVATE -Wall -Wextra)
add_test(NAME bwd COMMAND test_bwd)

# FFT против FFTW — только если есть fftw3.h + libfftw3f (test-only
# зависимость; сама библиотека от FFTW не зависит).
if(FFTW_INCLUDE_DIR AND FFTWF_LIBRARY)
  add_executable(bench_fft_fftw bench_fft_fftw.cpp)
  target_link_libraries(bench_fft_fftw PRIVATE rouge-numerics ${FFTWF_LIBRARY})
  target_include_directories(bench_fft_fftw PRIVATE ${FFTW_INCLUDE_DIR})
  target_compile_options(bench_fft_fftw PRIVATE -Wall -Wextra)
  add_test(NAME bench_fftw COMMAND bench_fft_fftw)
endif()

add_executable(test_shims test_shims.cpp)
target_link_libraries(test_shims PRIVATE rouge-cublas rouge-cudnn rouge-numerics)
target_compile_options(test_shims PRIVATE -Wall -Wextra)
add_test(NAME shims COMMAND test_shims)

# Замер против BLAS: OpenBLAS, если найден, иначе системный CBLAS (обычно
# reference BLAS — имя печатается, чтобы не выдать его за OpenBLAS). Если
# BLAS нет вовсе — только наивный пол. Скорость — отчёт, тест падает только
# на неверном результате.
add_executable(bench_gemm_blas bench_gemm_blas.cpp)
target_link_libraries(bench_gemm_blas PRIVATE rouge-numerics)
target_compile_options(bench_gemm_blas PRIVATE -Wall -Wextra)
if(BLAS_LIB AND BLAS_INC)
  target_link_libraries(bench_gemm_blas PRIVATE ${BLAS_LIB})
  target_include_directories(bench_gemm_blas PRIVATE ${BLAS_INC})
  target_compile_definitions(bench_gemm_blas PRIVATE
    ROUGE_HAVE_CBLAS=1 ROUGE_BLAS_NAME="${BLAS_NAME}")
endif()
add_test(NAME bench_blas COMMAND bench_gemm_blas)

# Замер свёртки против oneDNN (тот же im2col+свой GEMM против oneDNN).
if(DNNL_LIBRARY AND DNNL_INCLUDE_DIR)
  add_executable(bench_conv_dnnl bench_conv_dnnl.cpp)
  target_link_libraries(bench_conv_dnnl PRIVATE rouge-numerics ${DNNL_LIBRARY})
  target_include_directories(bench_conv_dnnl PRIVATE ${DNNL_INCLUDE_DIR})
  target_compile_options(bench_conv_dnnl PRIVATE -Wall -Wextra)
  add_test(NAME bench_dnnl COMMAND bench_conv_dnnl)
endif()
