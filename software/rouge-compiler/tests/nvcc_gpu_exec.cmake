# ROUGE-V: vendor-PTX end-to-end on a real NVIDIA GPU (stage 3).
#
# Where gpu_exec.cmake proves our canonical kernels run, this script proves a
# kernel written the way real device code is written runs too: real CUDA C is
# compiled by the vendor's own nvcc, the resulting PTX goes through ptx2ir,
# and the outcome executes on hardware bit-exactly against a host reference.
# The vendor PTX exercises what hand-written kernels do not: a mangled .entry
# name, __device__ globals addressed through mov/cvta, __syncthreads lowered
# through nvcc's own barrier form, and atomics every thread executes.
#
# Failure policy: same as gpu_exec.cmake —
#   * nvcc missing or rejecting the architecture -> SKIP (same policy as the
#     test_nvcc_ptx regression test, so the suite still runs without CUDA);
#   * NO-DEVICE / SKIP-OLD-ARCH from the runner   -> SKIP;
#   * anything else                               -> FATAL_ERROR.
#
# Required variables:
#   NVCC        nvcc executable (or "nvcc" to resolve from PATH)
#   CLANG       clang executable
#   PTX2IR      ptx2ir executable
#   RUNNER      gpu_exec_runreal binary
#   CU_SRC      real_nvcc.cu source file
#   WORK_DIR    scratch directory
#   SM          nvcc -arch target, e.g. "sm_75"

foreach(req NVCC CLANG PTX2IR RUNNER CU_SRC WORK_DIR SM)
  if(NOT DEFINED ${req} OR "${${req}}" STREQUAL "")
    message(STATUS "nvcc gpu exec: SKIP (missing -D${req}=...)")
    return()
  endif()
endforeach()

if(NOT EXISTS "${CU_SRC}")
  message(STATUS "nvcc gpu exec: SKIP (source not found: ${CU_SRC})")
  return()
endif()

file(MAKE_DIRECTORY "${WORK_DIR}")
set(vendor_ptx "${WORK_DIR}/real_nvcc.ptx")
set(ll "${WORK_DIR}/real_nvcc.ll")
set(ptx "${WORK_DIR}/real_nvcc_gpu.ptx")

execute_process(
  COMMAND "${NVCC}" -arch=${SM} -ptx "${CU_SRC}" -o "${vendor_ptx}"
  RESULT_VARIABLE rc
  OUTPUT_VARIABLE out
  ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
  message(STATUS "nvcc gpu exec: SKIP (nvcc unavailable or rejected -arch=${SM})")
  return()
endif()

execute_process(
  COMMAND "${PTX2IR}" --target nvptx64-nvidia-cuda "${vendor_ptx}" "${ll}"
  RESULT_VARIABLE rc
  OUTPUT_VARIABLE out
  ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "nvcc gpu exec: FAILED - ptx2ir rejected vendor PTX: ${err}")
endif()

execute_process(
  COMMAND "${CLANG}" --target=nvptx64-nvidia-cuda -march=${SM} -S
          "${ll}" -o "${ptx}"
  RESULT_VARIABLE rc
  OUTPUT_VARIABLE out
  ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "nvcc gpu exec: FAILED - clang could not lower vendor-PTX "
          "IR to PTX (exit ${rc}): ${err}")
endif()

execute_process(
  COMMAND "${RUNNER}" "${ptx}"
  RESULT_VARIABLE rc
  OUTPUT_VARIABLE out
  ERROR_VARIABLE err)
set(log "${out}\n${err}")
if(log MATCHES "NO-DEVICE|SKIP-OLD-ARCH")
  string(REGEX MATCH "NO-DEVICE[^\n]*|SKIP-OLD-ARCH[^\n]*" marker "${log}")
  message(STATUS "nvcc gpu exec: SKIP (no usable NVIDIA GPU) [${marker}]")
  return()
endif()
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "nvcc gpu exec: FAILED - (exit ${rc}):\n${log}")
endif()
message(STATUS "nvcc gpu exec: OK - vendor PTX executed bit-exactly on device")
