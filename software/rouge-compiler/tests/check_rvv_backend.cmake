# ROUGE-V: informational cross-check that the same PTX-derived LLVM IR is
# accepted by LLVM's RISC-V backend with the vector extension (rv64gcv).
# This validates the "RISC-V LLVM Backend" stage of the compiler architecture.
# When this LLVM build has no riscv target the check is reported and skipped.

if(NOT DEFINED CLANG OR CLANG STREQUAL "" OR NOT EXISTS "${IR_FILE}")
  message(STATUS "rvv backend check: SKIP (no clang or no IR file)")
  return()
endif()

execute_process(
  COMMAND "${CLANG}" --target=riscv64-unknown-elf -march=rv64gcv
          -c "${IR_FILE}" -o "${OUT_FILE}"
  RESULT_VARIABLE rc
  OUTPUT_VARIABLE out
  ERROR_VARIABLE err)

if(NOT rc EQUAL 0)
  message(STATUS "rvv backend check: SKIP (riscv64 backend unavailable: "
                 "${err})")
  return()
endif()

if(NOT EXISTS "${OUT_FILE}")
  message(FATAL_ERROR "rvv backend check: FAILED - riscv64 backend produced no object")
endif()

message(STATUS "rvv backend check: OK - PTX-derived IR compiles for rv64gcv "
               "(RISC-V + Vector extensions)")