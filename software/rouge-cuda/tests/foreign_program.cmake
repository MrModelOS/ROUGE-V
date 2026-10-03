# ROUGE-V Волна 1: сборка настоящей CUDA-программы и запуск её через шим.
#
# Три шага, каждый проверяется отдельно, и любой провал — наш баг:
#   1. nvcc -c      : объект с настоящей статической регистрацией ядер
#   2. nvcc -ptx    : тот же PTX, который получает рантайм
#   3. g++ + наш шим: исполняемый файл, НЕ требующий libcudart/libcuda
#
# Программа печатает "OK" только если все элементы сошлись.

foreach(req NVCC SRC WORK_DIR SHIM)
  if(NOT DEFINED ${req} OR "${${req}}" STREQUAL "")
    message(STATUS "foreign program: SKIP (missing -D${req}=...)")
    return()
  endif()
endforeach()

foreach(req NVCC SRC SHIM)
  if(NOT EXISTS "${${req}}")
    message(STATUS "foreign program: SKIP (${req} not found: ${${req}})")
    return()
  endif()
endforeach()

file(MAKE_DIRECTORY "${WORK_DIR}")

find_program(GXX NAMES g++ clang++ REQUIRED)

get_filename_component(SHIM_DIR "${SHIM}" DIRECTORY)
get_filename_component(SHIM_NAME "${SHIM}" NAME)

execute_process(
  COMMAND "${NVCC}" -arch=sm_75 -c "${SRC}" -o "${WORK_DIR}/prog.o"
  RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "foreign program: FAILED - nvcc -c exited ${rc}: ${err}")
endif()


# The link names NO CUDA library on purpose: every CUDA symbol must resolve from
# our own shim. With --no-as-needed the linker records the DT_NEEDED entry even
# though nothing is referenced yet, so the binary is loaded with the shim at
# startup and its symbols can satisfy the CUDA references (same trick as a
# libcuda.so.1 imposter dropped into LD_LIBRARY_PATH).
execute_process(
  COMMAND "${GXX}" -O2 "${WORK_DIR}/prog.o" -o "${WORK_DIR}/prog"
          -Wl,--no-as-needed -l:${SHIM_NAME} -L${SHIM_DIR} -Wl,-rpath,${SHIM_DIR}
  RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "foreign program: FAILED - link exited ${rc}: ${err}")
endif()

# Reference binary: the SAME source linked by nvcc against the real CUDA
# runtime. Comparing it with the shimmed binary is the actual claim: same
# program, two stacks, same output.
execute_process(
  COMMAND "${NVCC}" -arch=sm_75 "${SRC}" -o "${WORK_DIR}/prog_ref"
  RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "foreign program: FAILED - nvcc could not link the reference "
                      "binary (exit ${rc}): ${err}")
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -E env
          "LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}"
          "${CMAKE_CURRENT_BINARY_DIR}/test_foreign_program"
          "${WORK_DIR}/prog_ref" "${WORK_DIR}/prog" "${SHIM}"
  RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
string(REPLACE "\n" "\n    " pretty "${out}${err}")
message(STATUS "foreign program: ${pretty}")

if(NOT rc EQUAL 0)
  message(FATAL_ERROR "foreign program: FAILED - the foreign program did not run "
                      "without NVIDIA (exit ${rc})")
endif()