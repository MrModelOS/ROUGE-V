# ROUGE-V: the SEMANTIC check of a module in OUR dialect.
#
# The module is written in the `rouge` dialect (the launch, the thread id,
# the barrier, the shared memory). The script lowers it to loops, then
# through the standard MLIR -> LLVM IR -> binary chain, runs the program and
# compares what it PRINTED with the expected value.
#
# The structural checks (the pass produced a scf.for, a memref.alloc) say
# nothing about whether the kernel computes the right thing — this is the
# check that does.
#
# Required variables:
#   ROUGE_OPT      the rouge-opt driver
#   MLIR_OPT       mlir-opt
#   MLIR_TRANSLATE mlir-translate
#   CLANG          a C compiler
#   INPUT          the .mlir module
#   RUNTIME_C      the tiny print_f32 runtime
#   WORK_DIR       scratch directory
#   EXPECTED       the expected printed line (e.g. "sum=96.000000")
#   SIMT_PASS      the pass that lowers SIMT into loops (optional)

foreach(req ROUGE_OPT MLIR_OPT MLIR_TRANSLATE CLANG INPUT RUNTIME_C WORK_DIR EXPECTED)
  if(NOT DEFINED ${req} OR "${${req}}" STREQUAL "")
    message(STATUS "rouge semantics: SKIP (missing -D${req}=...)")
    return()
  endif()
endforeach()

if(NOT DEFINED SIMT_PASS OR "${SIMT_PASS}" STREQUAL "")
  set(SIMT_PASS --rouge-to-scf)
endif()

file(MAKE_DIRECTORY "${WORK_DIR}")
set(lowered_mlir "${WORK_DIR}/lowered.mlir")

execute_process(
  COMMAND "${ROUGE_OPT}" ${SIMT_PASS} "${INPUT}" -o "${lowered_mlir}"
  RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "rouge semantics: FAILED - rouge-opt exited ${rc}: ${err}")
endif()

execute_process(
  COMMAND "${MLIR_OPT}"
          --convert-scf-to-cf --convert-cf-to-llvm
          --convert-arith-to-llvm --convert-index-to-llvm
          --expand-strided-metadata --finalize-memref-to-llvm
          --convert-vector-to-llvm --convert-func-to-llvm
          --reconcile-unrealized-casts
          "${lowered_mlir}" -o "${WORK_DIR}/tmp.ll"
  RESULT_VARIABLE rc OUTPUT_VARIABLE o ERROR_VARIABLE e)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "rouge semantics: FAILED - mlir-opt exited ${rc}: ${e}")
endif()
execute_process(
  COMMAND "${MLIR_TRANSLATE}" --mlir-to-llvmir "${WORK_DIR}/tmp.ll" -o "${WORK_DIR}/out.ll"
  RESULT_VARIABLE rc OUTPUT_VARIABLE o ERROR_VARIABLE e)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "rouge semantics: FAILED - mlir-translate exited ${rc}: ${e}")
endif()
execute_process(
  COMMAND "${CLANG}" -O2 "${WORK_DIR}/out.ll" "${RUNTIME_C}" -o "${WORK_DIR}/prog"
  RESULT_VARIABLE rc OUTPUT_VARIABLE o ERROR_VARIABLE e)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "rouge semantics: FAILED - clang exited ${rc}: ${e}")
endif()

execute_process(
  COMMAND "${WORK_DIR}/prog"
  RESULT_VARIABLE rc OUTPUT_VARIABLE o ERROR_VARIABLE e)
string(STRIP "${o}" o)
message(STATUS "rouge semantics: the program printed '${o}'")
if(NOT o STREQUAL "${EXPECTED}")
  message(FATAL_ERROR
    "rouge semantics: FAILED - the program printed '${o}', expected "
    "'${EXPECTED}'")
endif()
message(STATUS "rouge semantics: OK - the kernel computed the expected value")
