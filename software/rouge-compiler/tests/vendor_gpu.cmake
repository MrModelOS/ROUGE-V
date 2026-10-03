# ROUGE-V: Tier-1 vendor-kernel sweep on a real NVIDIA GPU.
#
# For each vendor kernel (kernels/vendor/*.ptx — genuine nvcc output, not
# hand-written) this script
#   1. translates the ORIGINAL .ptx with our ptx2ir for the NVPTX backend,
#   2. lowers the IR to PTX with clang (-march=sm_75),
#   3. loads it through the CUDA Driver API and checks the result against a
#      host reference (bit-exact where the arithmetic order allows it).
#
# The multi-entry vendor modules need an explicit entry: ptx2ir --entry takes
# it, so each cell lists "kernel:entry".
#
# Failure policy (same as gpu_exec.cmake):
#   * no NVIDIA GPU / card older than sm_75 -> SKIP (exit 0, markers mapped
#     to SKIP by CTest);
#   * ptx2ir failure, backend rejection, device fault or wrong results ->
#     FATAL_ERROR. A wrong result after a successful module load is our bug.
#
# Required variables:
#   CLANG       clang executable
#   PTX2IR      ptx2ir executable
#   RUNNER      gpu_exec_vendor binary
#   KERNEL_DIR  directory holding kernels/ (vendor/ lives inside)
#   WORK_DIR    scratch directory for the generated .ll / .ptx files
#   CELLS       comma separated "kernel.entry" items
#   SM          target architecture for clang, e.g. "sm_75"

foreach(req CLANG PTX2IR RUNNER KERNEL_DIR WORK_DIR CELLS SM)
  if(NOT DEFINED ${req} OR "${${req}}" STREQUAL "")
    message(STATUS "vendor gpu exec: SKIP (missing -D${req}=...)")
    return()
  endif()
endforeach()

foreach(req CLANG PTX2IR RUNNER)
  if(NOT EXISTS "${${req}}")
    message(STATUS "vendor gpu exec: SKIP (${req} not found: ${${req}})")
    return()
  endif()
endforeach()

file(MAKE_DIRECTORY "${WORK_DIR}")
string(REPLACE "," ";" VENDOR_CELLS "${CELLS}")
set(ANY_RAN FALSE)

foreach(cell ${VENDOR_CELLS})
  string(REPLACE "." ";" parts "${cell}")
  list(LENGTH parts nparts)
  if(nparts LESS 2)
    message(FATAL_ERROR "vendor gpu exec: bad cell '${cell}', expected kernel.entry")
  endif()
  list(GET parts 0 kernel)
  list(GET parts 1 entry)

  set(src "${KERNEL_DIR}/vendor/${kernel}.ptx")
  if(NOT EXISTS "${src}")
    message(FATAL_ERROR "vendor gpu exec: FAILED - missing vendor kernel ${src}")
  endif()

  set(ll "${WORK_DIR}/${kernel}.ll")
  set(ptx "${WORK_DIR}/${kernel}.ptx")

  execute_process(
    COMMAND "${PTX2IR}" --target nvptx64-nvidia-cuda --entry "${entry}"
            "${src}" "${ll}"
    RESULT_VARIABLE rc
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "vendor gpu exec: FAILED - ${kernel}:${entry}: ptx2ir exited ${rc}: ${err}")
  endif()

  execute_process(
    COMMAND "${CLANG}" --target=nvptx64-nvidia-cuda -march=${SM} -O2 -S
            "${ll}" -o "${ptx}"
    RESULT_VARIABLE rc
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "vendor gpu exec: FAILED - ${kernel}:${entry}: clang could not lower "
                        "the IR to PTX (exit ${rc}): ${err}")
  endif()

  execute_process(
    COMMAND "${RUNNER}" "${KERNEL_DIR}/vendor" "${kernel}"
    RESULT_VARIABLE rc
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err)
  set(ANY_RAN TRUE)
  if(rc EQUAL 0 AND NOT (out MATCHES "NO-DEVICE" OR out MATCHES "SKIP-OLD-ARCH"))
    message(STATUS "vendor gpu exec: ${kernel}")
    string(REPLACE "\n" "\n    " pretty "${out}")
    message(STATUS "    ${pretty}")
  else()
    message(STATUS "vendor gpu exec: SKIP (${kernel}: ${out}${err})")
  endif()
endforeach()

if(NOT ANY_RAN)
  message(STATUS "vendor gpu exec: no cells ran")
endif()