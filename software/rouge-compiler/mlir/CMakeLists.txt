# ROUGE-V MLIR contour — stage 2 of the AOT pipeline (GPU -> Vector -> RVV).
#
# Enabled from the parent with -DROUGE_ENABLE_MLIR=ON. Requires an MLIR package
# (MLIR_DIR or an in-tree LLVM build): the host toolchain used for the
# PTX -> LLVM IR path ships clang without MLIR, so the contour is OFF by
# default and ctest simply has no MLIR tests to run.

# LLVM в собранном виде ссылается на сторонний CMake-пакет zstd
# (zstd::libzstd_static). Не everywhere есть статическая zstd — тогда
# подставляем общий интерфейс на shared-вариант: LLVMSupport использует zstd
# только для сжатых секций, shared-библиотеки достаточно.
find_package(zstd CONFIG QUIET)
if(zstd_FOUND AND NOT TARGET zstd::libzstd_static)
  add_library(zstd::libzstd_static INTERFACE IMPORTED)
  set_target_properties(zstd::libzstd_static PROPERTIES
    INTERFACE_LINK_LIBRARIES zstd::libzstd)
  message(STATUS "rouge MLIR contour: zstd::libzstd_static -> shared zstd")
endif()

find_package(MLIR CONFIG QUIET)

if(NOT MLIR_FOUND)
  message(STATUS "rouge MLIR contour: MLIR package not found — skipping")
  return()
endif()

message(STATUS "rouge MLIR contour: building against MLIR ${MLIR_VERSION}")

# ---- pass library ------------------------------------------------------
add_library(rouge-mlir-passes STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/Transforms/SimtAccessReport.cpp)
target_include_directories(rouge-mlir-passes PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS})
target_link_libraries(rouge-mlir-passes PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport)
target_compile_options(rouge-mlir-passes PRIVATE ${LLVM_COMPILE_OPTIONS})
if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(rouge-mlir-passes PUBLIC -fno-rtti)
endif()

# ---- rouge-opt driver --------------------------------------------------
add_executable(rouge-opt ${CMAKE_CURRENT_SOURCE_DIR}/tools/rouge-opt/rouge-opt.cpp)
target_link_libraries(rouge-opt PRIVATE
  rouge-mlir-passes
  MLIROptLib
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRFuncDialect
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRGPUDialect
  MLIRVectorDialect)

# ---- tests ------------------------------------------------------------
# Тест проверяет не только код возврата, но и СОДЕРЖИМОЕ отчёта: pass
# верифицирует модуль перед разбором, поэтому невалидный вход — жёсткая
# ошибка. Классификация (смешанный доступ / unit-stride) тоже проверяется.
add_test(NAME mlir_simt_access_report
  COMMAND rouge-opt --rouge-simt-access-report
          ${CMAKE_CURRENT_SOURCE_DIR}/test/simt_access_report.mlir)
set_tests_properties(mlir_simt_access_report PROPERTIES
  # Две независимые регулярки: порядок строк отчёта не гарантирован
  # (группировка по StringMap), поэтому «сначала mixed, потом unit-stride»
  # проверять нельзя.
  PASS_REGULAR_EXPRESSION
    "mixed access.*vectorize only the runs between barriers;unit-stride candidate.*SIMT->RVV vectorizable")

# Синтаксис memref.atomic_rmw изменился в MLIR 23 — отдельный вход и
# отдельная ячейка только для новых версий.
if(MLIR_VERSION_MAJOR GREATER_EQUAL 23)
  add_test(NAME mlir_simt_access_report_atomic
    COMMAND rouge-opt --rouge-simt-access-report
            ${CMAKE_CURRENT_SOURCE_DIR}/test/simt_access_report_atomic.mlir)
  set_tests_properties(mlir_simt_access_report_atomic PROPERTIES
    PASS_REGULAR_EXPRESSION "atomics=1")
endif()
