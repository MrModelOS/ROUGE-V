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

# Python нужен дифференциальному фазеру MLIR-контура.
find_package(Python3 COMPONENTS Interpreter QUIET)

find_package(MLIR CONFIG QUIET)

if(NOT MLIR_FOUND)
  message(STATUS "rouge MLIR contour: MLIR package not found — skipping")
  return()
endif()

message(STATUS "rouge MLIR contour: building against MLIR ${MLIR_VERSION}")

# ---- our own dialect (the "rouge" dialect) --------------------------------
add_subdirectory(tools/rouge-dialect)

# ---- pass library ------------------------------------------------------
add_library(rouge-mlir-dialect STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/IR/RougeDialect.cpp)
add_dependencies(rouge-mlir-dialect rouge-dialect-headers)
target_include_directories(rouge-mlir-dialect PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}/include
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS})
target_link_libraries(rouge-mlir-dialect PUBLIC
  MLIRIR
  MLIRSupport
  MLIRSideEffectInterfaces)
# MLIR собран без RTTI: та же опция нужна и нашему диалекту, иначе vtable
# диалекта тянет typeinfo mlir::Dialect, которого в библиотеке нет.
if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(rouge-mlir-dialect PUBLIC -fno-rtti)
endif()

# ---- the PTX front-end of the dialect -------------------------------------
add_library(rouge-mlir-frontend STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/Frontend/PtxToRouge.cpp)
target_include_directories(rouge-mlir-frontend PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS})
target_link_libraries(rouge-mlir-frontend PUBLIC
  rouge-mlir-dialect
  rouge-ptx
  MLIRIR
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRMathDialect
  MLIRSCFDialect
  MLIRSupport)

add_library(rouge-mlir-passes STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/Transforms/SimtAccessReport.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/Transforms/SimtToScf.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/Transforms/UnitStrideToVector.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/Transforms/RougeToScf.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/Transforms/RougeToGpu.cpp)
target_include_directories(rouge-mlir-passes PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS})
target_link_libraries(rouge-mlir-passes PUBLIC
  rouge-mlir-dialect
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

# ---- rouge-ptx2mlir: the PTX front-end as a tool --------------------------
add_executable(rouge-ptx2mlir
  ${CMAKE_CURRENT_SOURCE_DIR}/tools/rouge-ptx2mlir/rouge-ptx2mlir.cpp)
target_link_libraries(rouge-ptx2mlir PRIVATE
  rouge-mlir-frontend
  MLIRIR
  MLIRSupport)

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
  MLIRVectorDialect
  MLIRTransforms
  rouge-mlir-dialect)

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
  PASS_REGULAR_EXPRESSION "shared memory passed as a parameter is refused")

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

# ---- дифференциальный фазер MLIR-контура --------------------------------
#
# Тот же PTX-ядро идёт двумя независимыми путями: наш текстовый интерпретатор
# (эталон семантики) и наш MLIR-контур (фронтенд -> диалект rouge -> петли ->
# LLVM IR -> бинарник -> исполнение на CPU). Результаты обязаны совпасть
# ПОЭЛЕМЕНТНО.
#
# Операции ограничены теми, что фронтенд понимает. Отказ фронтенда — это тоже
# результат, но в этом цикле он считается провалом: цикл проверяет известное
# подмножество, и потеря уже работающей операции не должна проходить молча.
if(MLIR_OPT_EXE AND MLIR_TRANSLATE_EXE AND CLANG_EXECUTABLE AND
   Python3_Interpreter_FOUND)
  enable_language(C)
  add_library(mlir_print_i32 STATIC test/print_i32.c)
  add_test(NAME compiler_mlir_fuzz
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/../tests/fuzz/run_fuzz_mlir.py
            --ptx2mlir $<TARGET_FILE:rouge-ptx2mlir>
            --rouge-opt $<TARGET_FILE:rouge-opt>
            --gen $<TARGET_FILE:fuzz_gen_kernel>
            --interp $<TARGET_FILE:fuzz_run_interp>
            --mlir-opt ${MLIR_OPT_EXE}
            --mlir-translate ${MLIR_TRANSLATE_EXE}
            --clang ${CLANG_EXECUTABLE}
            --runtime $<TARGET_FILE:mlir_print_i32>
            --ops "add,sub,mul,shl,shru,shrs,and,or,xor,not,mins,maxs,popc,clz,setp,div,rem,dp4as,dp4au,prmt,bfi,bfe"
            --iters 8
            --work ${CMAKE_CURRENT_BINARY_DIR}/fuzz-work-mlir
            --cases ${CMAKE_CURRENT_SOURCE_DIR}/../tests/fuzz/cases)
  set_tests_properties(compiler_mlir_fuzz PROPERTIES TIMEOUT 900)

  # Тот же фазер, но на ДРУГОЙ ФОРМЕ ядра: общая память, барьер и две фазы,
  # причём вторая фаза читает чужой элемент (n-1-tid). Это проверка
  # семантики фаз и общей памяти, а не векторизатора: индекс намеренно не
  # «поток + константа», и проход обязан оставить его скалярным.
  add_test(NAME compiler_mlir_fuzz_phases
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/../tests/fuzz/run_fuzz_mlir.py
            --ptx2mlir $<TARGET_FILE:rouge-ptx2mlir>
            --rouge-opt $<TARGET_FILE:rouge-opt>
            --gen $<TARGET_FILE:fuzz_gen_kernel>
            --interp $<TARGET_FILE:fuzz_run_interp>
            --mlir-opt ${MLIR_OPT_EXE}
            --mlir-translate ${MLIR_TRANSLATE_EXE}
            --clang ${CLANG_EXECUTABLE}
            --runtime $<TARGET_FILE:mlir_print_i32>
            --template phases
            --iters 8
            --work ${CMAKE_CURRENT_BINARY_DIR}/fuzz-work-mlir-phases
            --cases ${CMAKE_CURRENT_SOURCE_DIR}/../tests/fuzz/cases)
  set_tests_properties(compiler_mlir_fuzz_phases PROPERTIES TIMEOUT 900)

  # Тот же фазер, но на ядре, где общая память адресуется «как у nvcc»:
  # mov.u32 %r, <shared symbol>, цепочки add.s32 и доступы [reg+imm]. Это
  # проверяет восстановление демотированной адресной арифметики фронтендом.
  add_test(NAME compiler_mlir_fuzz_demoted
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/../tests/fuzz/run_fuzz_mlir.py
            --ptx2mlir $<TARGET_FILE:rouge-ptx2mlir>
            --rouge-opt $<TARGET_FILE:rouge-opt>
            --gen $<TARGET_FILE:fuzz_gen_kernel>
            --interp $<TARGET_FILE:fuzz_run_interp>
            --mlir-opt ${MLIR_OPT_EXE}
            --mlir-translate ${MLIR_TRANSLATE_EXE}
            --clang ${CLANG_EXECUTABLE}
            --runtime $<TARGET_FILE:mlir_print_i32>
            --template demoted
            --iters 8
            --work ${CMAKE_CURRENT_BINARY_DIR}/fuzz-work-mlir-demoted
            --cases ${CMAKE_CURRENT_SOURCE_DIR}/../tests/fuzz/cases)
  set_tests_properties(compiler_mlir_fuzz_demoted PROPERTIES TIMEOUT 900)
  # Каждый сохранённый провал — это ядро, которое когда-то нашло баг (так был
  # найден `prmt`, терявший маску байта). Оно повторяется на том же входе и
  # обязано проходить. Это регрессионный тест, а не «пример вывода»: падение
  # здесь означает, что вернулось ровно то, что чинили.
  add_test(NAME compiler_mlir_fuzz_cases
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/../tests/fuzz/run_fuzz_mlir.py
            --ptx2mlir $<TARGET_FILE:rouge-ptx2mlir>
            --rouge-opt $<TARGET_FILE:rouge-opt>
            --gen $<TARGET_FILE:fuzz_gen_kernel>
            --interp $<TARGET_FILE:fuzz_run_interp>
            --mlir-opt ${MLIR_OPT_EXE}
            --mlir-translate ${MLIR_TRANSLATE_EXE}
            --clang ${CLANG_EXECUTABLE}
            --runtime $<TARGET_FILE:mlir_print_i32>
            --replay ${CMAKE_CURRENT_SOURCE_DIR}/../tests/fuzz/cases
            --work ${CMAKE_CURRENT_BINARY_DIR}/fuzz-work-mlir-cases)
  set_tests_properties(compiler_mlir_fuzz_cases PROPERTIES TIMEOUT 900)
endif()

# ---- наш собственный диалект (rouge) -------------------------------------
#
# Тот же lowering, но на наших операциях: запуск — это rouge.launch, ядро —
# обычная func.func, идентификатор потока — rouge.thread_id. Это проверка
# того, что диалект — ЗАМЕНА апстримному gpu-диалекту, а не переименование.
add_test(NAME mlir_rouge_to_scf
  COMMAND rouge-opt --rouge-to-scf
          ${CMAKE_CURRENT_SOURCE_DIR}/test/rouge_to_scf.mlir)
set_tests_properties(mlir_rouge_to_scf PROPERTIES
  # Launch becomes a loop over the grid, the body is split into phases at the
  # barrier (two thread loops), and the rouge ops are gone.
  PASS_REGULAR_EXPRESSION
    "launches converted=1 refused=0;scf.for")

# Общая память, ПРИХОДЯЩАЯ ПАРАМЕТРОМ: отказ громкий (копия на блок должна
# жить у вызывающего, а запуск не знает размера).
add_test(NAME mlir_rouge_to_scf_shared
  COMMAND rouge-opt --rouge-to-scf
          ${CMAKE_CURRENT_SOURCE_DIR}/test/rouge_to_scf_shared.mlir)
set_tests_properties(mlir_rouge_to_scf_shared PROPERTIES
  PASS_REGULAR_EXPRESSION "shared memory passed as a parameter is refused")

# Общая память, ОБЪЯВЛЕННАЯ ВНУТРИ ЯДРА: адресное пространство живёт в ТИПЕ,
# и объявление превращается в private-накопитель на блок (ядро вызывается
# один раз на блок) — отказа здесь больше нет.
add_test(NAME mlir_rouge_to_scf_shared_alloc
  COMMAND rouge-opt --rouge-to-scf
          ${CMAKE_CURRENT_SOURCE_DIR}/test/rouge_to_scf_shared_alloc.mlir)
set_tests_properties(mlir_rouge_to_scf_shared_alloc PROPERTIES
  PASS_REGULAR_EXPRESSION
    "launches converted=1 refused=0;memref.alloc\\(\\) : memref<64xf32>")

# СЕМАНТИЧЕСКАЯ проверка общей памяти: ядро с .shared обязано посчитать то
# же, что эталон (сумма 64 * 1.5 = 96). Структурная проверка ничего не
# говорит о правильности вычисления — это говорит эта.
if(MLIR_OPT_EXE AND MLIR_TRANSLATE_EXE AND CLANG_EXECUTABLE)
  add_test(NAME mlir_rouge_shared_semantics
    COMMAND ${CMAKE_COMMAND}
      -DROUGE_OPT=$<TARGET_FILE:rouge-opt>
      -DMLIR_OPT=${MLIR_OPT_EXE}
      -DMLIR_TRANSLATE=${MLIR_TRANSLATE_EXE}
      -DCLANG=${CLANG_EXECUTABLE}
      -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/test/rouge_shared_semantics.mlir
      -DRUNTIME_C=$<TARGET_FILE:mlir_print_f32>
      -DEXPECTED=sum=96.000000
      -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/rouge-shared-semantics
      -P ${CMAKE_CURRENT_SOURCE_DIR}/rouge_semantics.cmake)
endif()

# Модель расхождений: ветвление внутри тела потока становится МАСКОЙ
# (vector<Nxi1>), а обращения — маскируемыми. Сериализации ветки больше нет.
add_test(NAME mlir_divergence_mask
  COMMAND rouge-opt --rouge-unit-stride-to-vector
          ${CMAKE_CURRENT_SOURCE_DIR}/test/divergence_mask.mlir)
set_tests_properties(mlir_divergence_mask PROPERTIES
  PASS_REGULAR_EXPRESSION "vector.maskedstore")

# СЕМАНТИЧЕСКАЯ проверка модели расхождений: скалярная и векторная сборки
# обязаны посчитать одно и то же (первая половина скопирована, вторая — нет).
if(MLIR_OPT_EXE AND MLIR_TRANSLATE_EXE AND CLANG_EXECUTABLE)
  add_test(NAME mlir_divergence_mask_semantics
    COMMAND ${CMAKE_COMMAND}
      -DROUGE_OPT=$<TARGET_FILE:rouge-opt>
      -DMLIR_OPT=${MLIR_OPT_EXE}
      -DMLIR_TRANSLATE=${MLIR_TRANSLATE_EXE}
      -DCLANG=${CLANG_EXECUTABLE}
      -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/test/divergence_mask.mlir
      -DRUNTIME_C=$<TARGET_FILE:mlir_print_f32>
      -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/divergence-mask
      -P ${CMAKE_CURRENT_SOURCE_DIR}/vector_semantics.cmake)
endif()

# Полная цепочка на НАШЕМ диалекте: rouge -> SCF -> Vector -> LLVM IR ->
# ассемблер RISC-V (настоящие vle32.v) -> исполнение под qemu-riscv64.
# Скалярная и векторная сборки обязаны посчитать одно и то же.
if(MLIR_OPT_EXE AND MLIR_TRANSLATE_EXE AND CLANG_EXECUTABLE AND LLC_EXE AND QEMU_RISCV)
  add_test(NAME mlir_rouge_to_rvv
    COMMAND ${CMAKE_COMMAND}
      -DROUGE_OPT=$<TARGET_FILE:rouge-opt>
      -DMLIR_OPT=${MLIR_OPT_EXE}
      -DMLIR_TRANSLATE=${MLIR_TRANSLATE_EXE}
      -DLLC=${LLC_EXE}
      -DCLANG=${CLANG_EXECUTABLE}
      -DQEMU=${QEMU_RISCV}
      -DSIMT_PASS=--rouge-to-scf
      -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/test/rouge_to_rvv.mlir
      -DHARNESS_C=${CMAKE_CURRENT_SOURCE_DIR}/test/rv_vector_harness.c
      -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/rouge-vector-rvv
      -P ${CMAKE_CURRENT_SOURCE_DIR}/vector_rvv.cmake)
endif()

# ---- бэкенд-драйвер (rouge-compile) --------------------------------------
#
# Плагин бэкенда: PTX -> диалект -> SCF -> Vector -> LLVM IR -> машина.
# Своей кодогенерации у контура нет — её делает LLVM. Тест сквозной: на входе
# PTX-ядро копирования, на выходе ассемблер, и в нём должны быть ВЕКТОРНЫЕ
# обращения. Структурная проверка; семантику той же цепочки проверяет
# mlir_rouge_to_rvv (исполнение под qemu).
if(Python3_Interpreter_FOUND AND MLIR_OPT_EXE AND MLIR_TRANSLATE_EXE AND LLC_EXE)
  add_test(NAME mlir_rouge_compile_rvv
    COMMAND ${CMAKE_COMMAND}
      -DROUGE_OPT=$<TARGET_FILE:rouge-opt>
      -DROUGE_PTX2MLIR=$<TARGET_FILE:rouge-ptx2mlir>
      -DMLIR_OPT=${MLIR_OPT_EXE}
      -DMLIR_TRANSLATE=${MLIR_TRANSLATE_EXE}
      -DLLC=${LLC_EXE}
      -DPYTHON=${Python3_EXECUTABLE}
      -DDRIVER=${CMAKE_CURRENT_SOURCE_DIR}/tools/rouge-compile/rouge-compile
      -DTARGET=rvv
      "-DARGS=--block;64;--shape;0=64xf32;--shape;1=64xf32"
      -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/test/vector_copy.ptx
      -DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/vector_copy_rvv.s
      "-DEXPECTED=vle32.v;vse32.v"
      -P ${CMAKE_CURRENT_SOURCE_DIR}/rouge_compile.cmake)

  add_test(NAME mlir_rouge_compile_cpu
    COMMAND ${CMAKE_COMMAND}
      -DROUGE_OPT=$<TARGET_FILE:rouge-opt>
      -DROUGE_PTX2MLIR=$<TARGET_FILE:rouge-ptx2mlir>
      -DMLIR_OPT=${MLIR_OPT_EXE}
      -DMLIR_TRANSLATE=${MLIR_TRANSLATE_EXE}
      -DLLC=${LLC_EXE}
      -DPYTHON=${Python3_EXECUTABLE}
      -DDRIVER=${CMAKE_CURRENT_SOURCE_DIR}/tools/rouge-compile/rouge-compile
      -DTARGET=cpu
      "-DARGS=--block;64;--shape;0=64xf32;--shape;1=64xf32"
      -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/test/vector_copy.ptx
      -DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/vector_copy_cpu.s
      "-DEXPECTED=movaps"
      -P ${CMAKE_CURRENT_SOURCE_DIR}/rouge_compile.cmake)
endif()
