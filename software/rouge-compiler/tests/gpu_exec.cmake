# ROUGE-V: on-device execution check for the NVPTX backend (stage 2).
#
# Where gpu_targets.cmake only proves the IR *compiles* for a GPU triple, this
# script proves the compiled kernel *runs* on a real NVIDIA GPU and produces
# bit-exact results against a host reference. For every "kernel.selector" cell
# it
#   1. runs our own ptx2ir with --target nvptx64-nvidia-cuda;
#   2. lowers the IR to PTX with clang (-march=sm_75);
#   3. loads that PTX through the CUDA Driver API and checks the result.
#
# Failure policy:
#   * no NVIDIA GPU on this host, or a card older than sm_75 (the PTX target)
#     -> STATUS "SKIP", exit 0. The runner prints NO-DEVICE / SKIP-OLD-ARCH
#     markers; CTest also matches them via SKIP_REGULAR_EXPRESSION, so even a
#     nonzero exit on such a host is reported as skipped, never as failed;
#   * anything else (ptx2ir failed, clang rejected the IR, results mismatch,
#     device fault) -> FATAL_ERROR. A result mismatch after a passed module
#     load is our bug: the driver already accepted the code.
#
# Required variables:
#   CLANG       clang executable
#   PTX2IR      ptx2ir executable
#   RUNNER      gpu_exec_rungpu binary (built from gpu_exec_rungpu.c)
#   KERNEL_DIR  directory holding the canonical .ptx kernels
#   WORK_DIR    scratch directory for the generated .ll / .ptx files
#   CELLS       comma separated "kernel.selector" items,
#               e.g. "vadd.vadd,gemm_tile.gemm_tileB"
#   SM          target architecture for clang, e.g. "sm_75"

foreach(req CLANG PTX2IR RUNNER KERNEL_DIR WORK_DIR CELLS SM)
  if(NOT DEFINED ${req} OR "${${req}}" STREQUAL "")
    message(STATUS "gpu exec: SKIP (missing -D${req}=...)")
    return()
  endif()
endforeach()

foreach(req CLANG PTX2IR RUNNER)
  if(NOT EXISTS "${${req}}")
    message(STATUS "gpu exec: SKIP (${req} not found: ${${req}})")
    return()
  endif()
endforeach()

file(MAKE_DIRECTORY "${WORK_DIR}")
string(REPLACE "," ";" MATRIX_CELLS "${CELLS}")

function(rouge_exec_cell kernel selector)
  set(ll "${WORK_DIR}/${kernel}.ll")
  set(ptx "${WORK_DIR}/${kernel}.ptx")

  execute_process(
    COMMAND "${PTX2IR}" --target nvptx64-nvidia-cuda
            "${KERNEL_DIR}/${kernel}.ptx" "${ll}"
    RESULT_VARIABLE rc
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "gpu exec: FAILED - ${kernel}: ptx2ir exited ${rc}: ${err}")
  endif()

  execute_process(
    COMMAND "${CLANG}" --target=nvptx64-nvidia-cuda -march=${SM} -S
            "${ll}" -o "${ptx}"
    RESULT_VARIABLE rc
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "gpu exec: FAILED - ${kernel}: clang could not lower "
            "the IR to PTX (exit ${rc}): ${err}")
  endif()

  execute_process(
    COMMAND "${RUNNER}" "${ptx}" "${selector}"
    RESULT_VARIABLE rc
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err)
  set(log "${out}\n${err}")
  if(log MATCHES "NO-DEVICE|SKIP-OLD-ARCH")
    # Re-emit the runner's marker into our own output: CTest matches
    # SKIP_REGULAR_EXPRESSION against what this script prints, not against the
    # runner log we consumed into a variable.
    string(REGEX MATCH "NO-DEVICE[^\n]*|SKIP-OLD-ARCH[^\n]*" marker "${log}")
    message(STATUS "gpu exec: SKIP ${kernel}.${selector} (no usable NVIDIA GPU "
                   "on this host) [${marker}]")
    set(ROUGE_EXEC_SKIP 1 PARENT_SCOPE)
    return()
  endif()
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "gpu exec: FAILED - ${kernel}.${selector} (exit ${rc}):\n${log}")
  endif()
  message(STATUS "gpu exec: OK ${kernel}.${selector} (bit-exact on device)")
  set(ROUGE_EXEC_OK 1 PARENT_SCOPE)
endfunction()

set(ROUGE_EXEC_OK 0)
set(ROUGE_EXEC_SKIP 0)
foreach(cell IN LISTS MATRIX_CELLS)
  string(REPLACE "." ";" parts "${cell}")
  list(LENGTH parts nparts)
  if(NOT nparts EQUAL 2)
    message(FATAL_ERROR "gpu exec: FAILED - malformed cell \"${cell}\", "
                        "expected \"kernel.selector\"")
  endif()
  list(GET parts 0 kernel)
  list(GET parts 1 selector)
  if(NOT EXISTS "${KERNEL_DIR}/${kernel}.ptx")
    message(FATAL_ERROR "gpu exec: FAILED - missing kernel ${KERNEL_DIR}/${kernel}.ptx")
  endif()
  rouge_exec_cell("${kernel}" "${selector}")
endforeach()

if(ROUGE_EXEC_SKIP AND NOT ROUGE_EXEC_OK)
  message(STATUS "gpu exec: SKIP - no kernel was exercised on this host")
endif()
if(ROUGE_EXEC_OK)
  message(STATUS "gpu exec: OK - device execution verified bit-exact")
endif()
