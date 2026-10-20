# ROUGE-V: the SEMANTIC check of the unit-stride -> vector pass.
#
# The structural check (the pass produced a vector.load) says nothing about
# whether the vectorised kernel computes the same thing. This script runs the
# module TWICE through the standard MLIR -> LLVM IR -> binary chain:
#
#   1) as written (the scalar kernel);
#   2) after --rouge-unit-stride-to-vector (the vectorised kernel);
#
# and compares what the two programs PRINT. The kernel copies a known
# pattern, so the sums must agree bit for bit.
#
# Required variables:
#   ROUGE_OPT      the rouge-opt driver
#   MLIR_OPT       mlir-opt
#   MLIR_TRANSLATE mlir-translate
#   CLANG          a C compiler
#   INPUT          the .mlir module
#   RUNTIME_C      the tiny print_f32 runtime
#   WORK_DIR       scratch directory

foreach(req ROUGE_OPT MLIR_OPT MLIR_TRANSLATE CLANG INPUT RUNTIME_C WORK_DIR)
  if(NOT DEFINED ${req} OR "${${req}}" STREQUAL "")
    message(STATUS "vector semantics: SKIP (missing -D${req}=...)")
    return()
  endif()
endforeach()

file(MAKE_DIRECTORY "${WORK_DIR}")
set(scalar_ll "${WORK_DIR}/scalar.ll")
set(scalar_bin "${WORK_DIR}/scalar")
set(vec_mlir "${WORK_DIR}/vectorised.mlir")
set(vec_ll "${WORK_DIR}/vectorised.ll")
set(vec_bin "${WORK_DIR}/vectorised")

# 1) the vectorised module
execute_process(
  COMMAND "${ROUGE_OPT}" --rouge-unit-stride-to-vector "${INPUT}"
          -o "${vec_mlir}"
  RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "vector semantics: FAILED - rouge-opt exited ${rc}: ${err}")
endif()
if(NOT EXISTS "${vec_mlir}")
  message(FATAL_ERROR "vector semantics: FAILED - rouge-opt produced no module")
endif()

# 2) MLIR -> LLVM IR -> binary, for both modules
function(rouge_lower_and_run in_mlir out_ll out_bin out_text)
  execute_process(
    COMMAND "${MLIR_OPT}"
            --convert-scf-to-cf --convert-cf-to-llvm
            --convert-arith-to-llvm --convert-index-to-llvm
            --expand-strided-metadata --finalize-memref-to-llvm
            --convert-vector-to-llvm --convert-func-to-llvm
            --reconcile-unrealized-casts
            "${in_mlir}" -o "${WORK_DIR}/tmp.ll"
    RESULT_VARIABLE rc OUTPUT_VARIABLE o ERROR_VARIABLE e)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "vector semantics: FAILED - mlir-opt exited ${rc}: ${e}")
  endif()
  execute_process(
    COMMAND "${MLIR_TRANSLATE}" --mlir-to-llvmir "${WORK_DIR}/tmp.ll" -o "${out_ll}"
    RESULT_VARIABLE rc OUTPUT_VARIABLE o ERROR_VARIABLE e)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "vector semantics: FAILED - mlir-translate exited ${rc}: ${e}")
  endif()
  execute_process(
    COMMAND "${CLANG}" -O2 "${out_ll}" "${RUNTIME_C}" -o "${out_bin}"
    RESULT_VARIABLE rc OUTPUT_VARIABLE o ERROR_VARIABLE e)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "vector semantics: FAILED - clang exited ${rc}: ${e}")
  endif()
  # The exit code of the lowered program is not the point (the lowered
  # main() does not necessarily return 0); the PRINTED value is. A crash
  # would show up as a missing or garbled line, which the comparison
  # below catches.
  execute_process(
    COMMAND "${out_bin}"
    RESULT_VARIABLE rc OUTPUT_VARIABLE o ERROR_VARIABLE e)
  if(o STREQUAL "")
    message(FATAL_ERROR "vector semantics: FAILED - the program printed "
                        "nothing (exit ${rc}): ${e}")
  endif()
  set(${out_text} "${o}" PARENT_SCOPE)
endfunction()

rouge_lower_and_run("${INPUT}" "${scalar_ll}" "${scalar_bin}" scalar_text)
rouge_lower_and_run("${vec_mlir}" "${vec_ll}" "${vec_bin}" vec_text)

string(STRIP "${scalar_text}" scalar_text)
string(STRIP "${vec_text}" vec_text)
message(STATUS "vector semantics: scalar   -> ${scalar_text}")
message(STATUS "vector semantics: vectorised -> ${vec_text}")
if(NOT scalar_text STREQUAL vec_text)
  message(FATAL_ERROR
    "vector semantics: FAILED - the vectorised kernel computed '${vec_text}' "
    "but the scalar one computed '${scalar_text}'")
endif()
message(STATUS "vector semantics: OK - the vectorised kernel computes the same")
