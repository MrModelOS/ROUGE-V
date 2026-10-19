# ROUGE-V Волна 1: vendor CUDA program that uses the REAL cuBLAS.
#
# The program (programs/cublas_gemm.cu) contains no CUDA kernel at all - only
# a cuBLAS call. It is built twice from the SAME object:
#   1. reference: linked by nvcc against the real libcudart + libcublas, run
#      on the real NVIDIA GPU (the numbers come from NVIDIA's own library);
#   2. shimmed : linked by g++ against OUR shims only (no CUDA library is
#      named at all - every cuda*/cu*/cublas* symbol must resolve from our
#      own code), run on the CPU with emulation forced.
#
# That is the vendor-lock-in test that matters: not "our own kernel runs",
# but "a program linked against NVIDIA's proprietary BLAS computes on a
# machine without NVIDIA".
#
# Needs nvcc for compiling/linking the reference. Without nvcc the test
# reports SKIP. The reference run needs a real GPU; without one the test
# says so honestly instead of pretending.

foreach(req NVCC PROGRAM WORK_DIR SHIM CUBLAS_SHIM TEST_BIN)
  if(NOT DEFINED ${req} OR "${${req}}" STREQUAL "")
    message(STATUS "foreign cublas program: SKIP (missing -D${req}=...)")
    return()
  endif()
endforeach()

foreach(req NVCC SHIM CUBLAS_SHIM TEST_BIN)
  if(NOT EXISTS "${${req}}")
    message(STATUS "foreign cublas program: SKIP (${req} not found: ${${req}})")
    return()
  endif()
endforeach()

file(MAKE_DIRECTORY "${WORK_DIR}")
find_program(GXX NAMES g++ clang++ REQUIRED)

get_filename_component(name "${PROGRAM}" NAME_WE)
set(obj "${WORK_DIR}/${name}.o")
set(prog "${WORK_DIR}/${name}")
set(prog_ref "${WORK_DIR}/${name}_ref")

# 1. Compile the vendor source with the vendor compiler.
execute_process(COMMAND "${NVCC}" -arch=sm_75 -c "${PROGRAM}" -o "${obj}"
  RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "foreign cublas program: FAILED - ${name}: nvcc -c exited ${rc}: ${err}")
endif()

# 2. Link the SAME object against our shims only. No CUDA library is named:
#    every cuda*/cu*/cublas* symbol has to resolve from our own code.
get_filename_component(SHIM_DIR "${SHIM}" DIRECTORY)
get_filename_component(SHIM_NAME "${SHIM}" NAME)
get_filename_component(CUBLAS_DIR "${CUBLAS_SHIM}" DIRECTORY)
get_filename_component(CUBLAS_NAME "${CUBLAS_SHIM}" NAME)
execute_process(
  COMMAND "${GXX}" -O2 "${obj}" -o "${prog}"
          -Wl,--no-as-needed
          -l:${SHIM_NAME} -L${SHIM_DIR} -Wl,-rpath,${SHIM_DIR}
          -l:${CUBLAS_NAME} -L${CUBLAS_DIR} -Wl,-rpath,${CUBLAS_DIR}
  RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "foreign cublas program: FAILED - ${name}: link against our shims exited ${rc}: ${err}")
endif()

# 3. Reference: the same program against the real cudart + real cuBLAS.
execute_process(COMMAND "${NVCC}" -arch=sm_75 "${PROGRAM}" -o "${prog_ref}" -lcublas
  RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "foreign cublas program: FAILED - ${name}: nvcc could not link the "
                      "reference binary against the real cuBLAS (exit ${rc}): ${err}")
endif()

# 4. Run both and compare.
execute_process(
  COMMAND ${CMAKE_COMMAND} -E env
          "LD_LIBRARY_PATH=${SHIM_DIR}:${CUBLAS_DIR}" "ROUGE_CUDA_FORCE_EMU=1"
          "${TEST_BIN}" "${prog_ref}" "${prog}"
  RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
string(REPLACE "\n" "\n    " pretty "${out}${err}")
message(STATUS "foreign cublas program [${name}]: ${pretty}")
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "foreign cublas program: FAILED - ${name} did not run on our shim "
                      "(exit ${rc})")
endif()
