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
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/Transforms/SimtAccessReport.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/Transforms/SimtToScf.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/Transforms/UnitStrideToVector.cpp)
target_include_directories(rouge-mlir-passes PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS})
target_link_libraries(rouge-mlir-passes PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRVectorDialect)
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
  MLIRControlFlowDialect
  MLIRMemRefDialect
  MLIRSCFDialect
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

# SIMT -> SCF: thread/block ids become explicit loops, the body is split
# into phases at barriers, the launch becomes a loop over the grid.
add_test(NAME mlir_simt_to_scf
  COMMAND rouge-opt --rouge-simt-to-scf
          ${CMAKE_CURRENT_SOURCE_DIR}/test/simt_to_scf.mlir)
set_tests_properties(mlir_simt_to_scf PROPERTIES
  # The kernel must become two thread loops (the barrier splits the body into
  # phases), the launch must become a grid loop, and gpu.thread_id must be
  # gone (replaced by the induction variable).
  PASS_REGULAR_EXPRESSION
    "kernels converted=1 refused=0;scf.for")

# A divergent branch (no barrier inside) is NOT an error: the threads run
# sequentially, so each takes its own path. The kernel must be lowered.
add_test(NAME mlir_simt_to_scf_divergent
  COMMAND rouge-opt --rouge-simt-to-scf
          ${CMAKE_CURRENT_SOURCE_DIR}/test/simt_to_scf_divergent.mlir)
set_tests_properties(mlir_simt_to_scf_divergent PROPERTIES
  PASS_REGULAR_EXPRESSION "kernels converted=1 refused=0")

# A BARRIER inside a divergent branch breaks the gpu.barrier contract, so the
# result would be undefined — the pass must refuse loudly.
add_test(NAME mlir_simt_to_scf_barrier_divergence
  COMMAND rouge-opt --rouge-simt-to-scf
          ${CMAKE_CURRENT_SOURCE_DIR}/test/simt_to_scf_barrier_divergence.mlir)
set_tests_properties(mlir_simt_to_scf_barrier_divergence PROPERTIES
  PASS_REGULAR_EXPRESSION "barrier inside a divergent branch")

# Ядро с общей памятью: memref с непустым адресным пространством. Копия
# общей памяти на блок была бы потеряна молча — проход обязан отказать.
add_test(NAME mlir_simt_to_scf_shared
  COMMAND rouge-opt --rouge-simt-to-scf
          ${CMAKE_CURRENT_SOURCE_DIR}/test/simt_to_scf_shared.mlir)
set_tests_properties(mlir_simt_to_scf_shared PROPERTIES
  PASS_REGULAR_EXPRESSION "using shared memory is refused")

# Векторизация unit-stride доступов: структурная проверка + СЕМАНТИЧЕСКАЯ
# (скалярная и векторная версии обязаны посчитать одно и то же).
add_test(NAME mlir_unit_stride_to_vector
  COMMAND rouge-opt --rouge-unit-stride-to-vector
          ${CMAKE_CURRENT_SOURCE_DIR}/test/unit_stride_to_vector.mlir)
set_tests_properties(mlir_unit_stride_to_vector PROPERTIES
  PASS_REGULAR_EXPRESSION
    "loops vectorised=1 refused=0;vector.load")

# The semantic check needs the MLIR tools (mlir-opt / mlir-translate) and a C
# compiler; without them there is nothing to compare.
get_filename_component(MLIR_BIN_DIR "${MLIR_DIR}/../../../bin" ABSOLUTE)
find_program(MLIR_OPT_EXE mlir-opt HINTS ${MLIR_BIN_DIR})
find_program(MLIR_TRANSLATE_EXE mlir-translate HINTS ${MLIR_BIN_DIR})
if(MLIR_OPT_EXE AND MLIR_TRANSLATE_EXE AND CLANG_EXECUTABLE)
  enable_language(C)
  add_library(mlir_print_f32 STATIC test/print_f32.c)
  add_test(NAME mlir_unit_stride_to_vector_semantics
    COMMAND ${CMAKE_COMMAND}
      -DROUGE_OPT=$<TARGET_FILE:rouge-opt>
      -DMLIR_OPT=${MLIR_OPT_EXE}
      -DMLIR_TRANSLATE=${MLIR_TRANSLATE_EXE}
      -DCLANG=${CLANG_EXECUTABLE}
      -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/test/unit_stride_to_vector.mlir
      -DRUNTIME_C=$<TARGET_FILE:mlir_print_f32>
      -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/vector-semantics
      -P ${CMAKE_CURRENT_SOURCE_DIR}/vector_semantics.cmake)
endif()

# Полная цепочка SIMT -> SCF -> Vector -> RVV: модуль векторизуется, lowers
# в LLVM IR, собирается llc -march=rv64gcv и исполняется под qemu-riscv64.
# Скалярная версия того же модуля собирается и запускается так же — результаты
# должны совпасть, а в ассемблере векторной сборки должны быть настоящие
# инструкции RVV (vle32.v / vse32.v).
find_program(LLC_EXE llc HINTS ${MLIR_BIN_DIR})
find_program(QEMU_RISCV qemu-riscv64
  HINTS $ENV{HOME}/.local/pkgs/qemu-user-11.1.2-1-x86_64/usr/bin
        /usr/local/bin /usr/bin)
if(MLIR_OPT_EXE AND MLIR_TRANSLATE_EXE AND CLANG_EXECUTABLE AND LLC_EXE AND QEMU_RISCV)
  add_test(NAME mlir_unit_stride_to_vector_rvv
    COMMAND ${CMAKE_COMMAND}
      -DROUGE_OPT=$<TARGET_FILE:rouge-opt>
      -DMLIR_OPT=${MLIR_OPT_EXE}
      -DMLIR_TRANSLATE=${MLIR_TRANSLATE_EXE}
      -DLLC=${LLC_EXE}
      -DCLANG=${CLANG_EXECUTABLE}
      -DQEMU=${QEMU_RISCV}
      -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/test/simt_to_rvv.mlir
      -DHARNESS_C=${CMAKE_CURRENT_SOURCE_DIR}/test/rv_vector_harness.c
      -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/vector-rvv
      -P ${CMAKE_CURRENT_SOURCE_DIR}/vector_rvv.cmake)
endif()

# Синтаксис memref.atomic_rmw изменился в MLIR 23 — отдельный вход и
# отдельная ячейка только для новых версий.
if(MLIR_VERSION_MAJOR GREATER_EQUAL 23)
  add_test(NAME mlir_simt_access_report_atomic
    COMMAND rouge-opt --rouge-simt-access-report
            ${CMAKE_CURRENT_SOURCE_DIR}/test/simt_access_report_atomic.mlir)
  set_tests_properties(mlir_simt_access_report_atomic PROPERTIES
    PASS_REGULAR_EXPRESSION "atomics=1")
endif()
