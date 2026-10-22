# ROUGE-V: our own MLIR contour, executed on the card.
#
# Where gpu_exec.cmake drives the AOT path (ptx2ir -> clang -> PTX), this one
# drives the DIALECT path:
#
#     PTX --rouge-ptx2mlir--> rouge --rouge-to-gpu--> gpu
#         --convert-gpu-to-nvvm--> LLVM/NVVM --rouge-extract-kernels-->
#         LLVM IR --clang (NVPTX)--> .ptx --CUDA driver--> the card
#
# and checks the result with the same runner and the same selector the AOT
# path uses. If the two paths ever disagree, this cell and the AOT cell fail
# differently — which is exactly what we want from two independent
# implementations.
#
# The .ptx must keep the PTX ABI: the memref arguments become bare pointers
# (--convert-gpu-to-nvvm=use-bare-ptr-memref-call-conv), so the runner passes
# plain device pointers. That is also why every memref must have a STATIC
# shape: the bare-pointer ABI has nowhere to carry the extent.
#
# Variables:
#   ROUGE_COMPILE   the backend driver (tools/rouge-compile/rouge-compile)
#   ROUGE_OPT       rouge-opt
#   PTX2MLIR        rouge-ptx2mlir
#   MLIR_OPT        the full mlir-opt (the gpu -> LLVM passes live there)
#   MLIR_TRANSLATE  mlir-translate
#   CLANG           clang with the NVPTX backend
#   RUNNER          gpu_exec_rungpu
#   KERNEL          the .ptx to translate
#   SELECTOR        the runner's check for this kernel
#   BLOCK           the block size the fuzzer/harness uses
#   SHAPES          comma-separated "index=shape" hints for the front-end
#   SM              the NVPTX architecture
#   WORK_DIR        scratch directory

foreach(req ROUGE_COMPILE ROUGE_OPT PTX2MLIR MLIR_OPT MLIR_TRANSLATE CLANG
            RUNNER KERNEL SELECTOR BLOCK SHAPES WORK_DIR SM)
  if(NOT DEFINED ${req} OR "${${req}}" STREQUAL "")
    message(STATUS "mlir gpu exec: SKIP (missing -D${req}=...)")
    return()
  endif()
endforeach()

foreach(req ROUGE_COMPILE ROUGE_OPT PTX2MLIR MLIR_OPT MLIR_TRANSLATE CLANG
            RUNNER KERNEL)
  if(NOT EXISTS "${${req}}")
    message(STATUS "mlir gpu exec: SKIP (${req} not found: ${${req}})")
    return()
  endif()
endforeach()

file(MAKE_DIRECTORY "${WORK_DIR}")
set(ptx "${WORK_DIR}/${SELECTOR}_mlir.ptx")

set(shape_args "")
string(REPLACE "," ";" SHAPE_LIST "${SHAPES}")
foreach(shape IN LISTS SHAPE_LIST)
  list(APPEND shape_args "--shape" "${shape}")
endforeach()

execute_process(
  COMMAND "${ROUGE_COMPILE}" "${KERNEL}" --target nvptx --sm "${SM}"
          --block "${BLOCK}" ${shape_args} -o "${ptx}"
          --rouge-opt "${ROUGE_OPT}" --ptx2mlir "${PTX2MLIR}"
          --mlir-opt "${MLIR_OPT}" --mlir-translate "${MLIR_TRANSLATE}"
          --clang "${CLANG}"
  RESULT_VARIABLE rc
  OUTPUT_VARIABLE out
  ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "mlir gpu exec: FAILED - the MLIR contour did not "
          "produce a .ptx (exit ${rc}):\n${out}\n${err}")
endif()

execute_process(
  COMMAND "${RUNNER}" "${ptx}" "${SELECTOR}"
  RESULT_VARIABLE rc
  OUTPUT_VARIABLE out
  ERROR_VARIABLE err)
set(log "${out}\n${err}")
if(log MATCHES "NO-DEVICE|SKIP-OLD-ARCH")
  string(REGEX MATCH "NO-DEVICE[^\n]*|SKIP-OLD-ARCH[^\n]*" marker "${log}")
  message(STATUS "mlir gpu exec: SKIP (no usable NVIDIA GPU on this host) "
                 "[${marker}]")
  return()
endif()
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "mlir gpu exec: FAILED - ${SELECTOR} (exit ${rc}):\n${log}")
endif()
message(STATUS "mlir gpu exec: OK ${SELECTOR} (the MLIR contour's .ptx is "
               "bit-exact on the card)")
