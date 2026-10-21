# ROUGE-V: the full SIMT -> SCF -> Vector -> RVV chain, executed.
#
# This closes the loop of the contour:
#
#   MLIR (SIMT)  --rouge-simt-to-scf-->  loops over the thread index
#                --rouge-unit-stride-to-vector-->  vector loop + scalar tail
#                --MLIR to LLVM IR-->  llc -march=rv64gcv  -->  RVV assembly
#                --link + run under qemu-riscv64-->  the number
#
# The scalar version of the same module is built and run the same way, and
# the two results must be EQUAL. The vectorised build must also contain real
# RISC-V vector instructions (vle32.v) — otherwise the "RVV" in the title
# would be a lie.
#
# Required variables:
#   ROUGE_OPT      the rouge-opt driver
#   MLIR_OPT       mlir-opt
#   MLIR_TRANSLATE mlir-translate
#   LLC            llc
#   CLANG          a C compiler
#   QEMU           qemu-riscv64
#   INPUT          the .mlir module
#   SIMT_PASS      the pass that lowers SIMT into loops; by default the
#                  upstream gpu dialect's one (--rouge-simt-to-scf), and
#                  --rouge-to-scf for OUR dialect
#   HARNESS_C      the freestanding RISC-V harness (C source; it is compiled
#                  for riscv64 HERE, not for the host — the register names
#                  and the exit syscall are RISC-V ones)
#   WORK_DIR       scratch directory

foreach(req ROUGE_OPT MLIR_OPT MLIR_TRANSLATE LLC CLANG QEMU INPUT HARNESS_C WORK_DIR)
  if(NOT DEFINED ${req} OR "${${req}}" STREQUAL "")
    message(STATUS "vector rvv: SKIP (missing -D${req}=...)")
    return()
  endif()
endforeach()

file(MAKE_DIRECTORY "${WORK_DIR}")

# The pass that lowers the SIMT structure into loops: the upstream gpu
# dialect's one, or ours.
if(NOT DEFINED SIMT_PASS OR "${SIMT_PASS}" STREQUAL "")
  set(SIMT_PASS --rouge-simt-to-scf)
endif()

# 1) The SIMT module -> SCF (the scalar reference) and -> SCF -> Vector.
set(scf_mlir "${WORK_DIR}/scf.mlir")
set(vec_mlir "${WORK_DIR}/vectorised.mlir")
execute_process(
  COMMAND "${ROUGE_OPT}" ${SIMT_PASS} "${INPUT}" -o "${scf_mlir}"
  RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "vector rvv: FAILED - rouge-opt (scf) exited ${rc}: ${err}")
endif()
execute_process(
  COMMAND "${ROUGE_OPT}" ${SIMT_PASS} --rouge-unit-stride-to-vector
          "${INPUT}" -o "${vec_mlir}"
  RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "vector rvv: FAILED - rouge-opt exited ${rc}: ${err}")
endif()

# 2) MLIR -> LLVM IR (the same chain as the x86 semantic check).
function(rouge_to_llvm in_mlir out_ll)
  execute_process(
    COMMAND "${MLIR_OPT}"
            --convert-scf-to-cf --convert-cf-to-llvm
            --convert-arith-to-llvm --convert-index-to-llvm
            --expand-strided-metadata --finalize-memref-to-llvm
            --convert-math-to-llvm --convert-vector-to-llvm --convert-func-to-llvm
            --reconcile-unrealized-casts
            "${in_mlir}" -o "${WORK_DIR}/tmp.ll"
    RESULT_VARIABLE rc OUTPUT_VARIABLE o ERROR_VARIABLE e)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "vector rvv: FAILED - mlir-opt exited ${rc}: ${e}")
  endif()
  execute_process(
    COMMAND "${MLIR_TRANSLATE}" --mlir-to-llvmir "${WORK_DIR}/tmp.ll" -o "${out_ll}"
    RESULT_VARIABLE rc OUTPUT_VARIABLE o ERROR_VARIABLE e)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "vector rvv: FAILED - mlir-translate exited ${rc}: ${e}")
  endif()
endfunction()

# 3) LLVM IR -> RISC-V assembly, link, run. The exit code is the result.
function(rouge_run_rv in_mlir tag out_code)
  rouge_to_llvm("${in_mlir}" "${WORK_DIR}/${tag}.ll")
  execute_process(
    COMMAND "${LLC}" -mtriple=riscv64-unknown-linux-gnu
            -mattr=+v,+m,+a,+f,+d,+c -O2
            "${WORK_DIR}/${tag}.ll" -o "${WORK_DIR}/${tag}.s"
    RESULT_VARIABLE rc OUTPUT_VARIABLE o ERROR_VARIABLE e)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "vector rvv: FAILED - llc exited ${rc}: ${e}")
  endif()
  execute_process(
    COMMAND "${CLANG}" --target=riscv64-unknown-linux-gnu -march=rv64gcv
            -O2 -ffreestanding -nostdlib -static -fuse-ld=lld
            "${WORK_DIR}/${tag}.s" "${HARNESS_C}" -o "${WORK_DIR}/${tag}.bin"
    RESULT_VARIABLE rc OUTPUT_VARIABLE o ERROR_VARIABLE e)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "vector rvv: FAILED - cannot link the riscv64 binary: ${e}")
  endif()
  execute_process(
    COMMAND "${QEMU}" "${WORK_DIR}/${tag}.bin"
    RESULT_VARIABLE rc OUTPUT_VARIABLE o ERROR_VARIABLE e)
  set(${out_code} "${rc}" PARENT_SCOPE)
endfunction()

rouge_run_rv("${scf_mlir}" scalar scalar_code)
rouge_run_rv("${vec_mlir}" vectorised vector_code)
message(STATUS "vector rvv: scalar exit=${scalar_code}, vectorised exit=${vector_code}")

if(NOT scalar_code STREQUAL vector_code)
  message(FATAL_ERROR
    "vector rvv: FAILED - the vectorised module exited ${vector_code}, the "
    "scalar one ${scalar_code}")
endif()

# 4) The vectorised build must really contain RVV instructions.
file(READ "${WORK_DIR}/vectorised.s" asm_text)
if(NOT asm_text MATCHES "vle32\\.v" AND NOT asm_text MATCHES "vse32\\.v")
  message(FATAL_ERROR
    "vector rvv: FAILED - the vectorised build has no RISC-V vector "
    "instructions (no vle32.v / vse32.v)")
endif()
message(STATUS "vector rvv: OK - the vectorised kernel ran on rv64gcv under "
                "qemu-user and computed the same value as the scalar one")
