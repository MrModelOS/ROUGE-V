# ROUGE-V Волна 1: сборка настоящих CUDA-программ и запуск их через шим.
#
# Для каждой программы в programs/ проверяется одно и то же:
#   1. nvcc -c      : объект с настоящей статической регистрацией ядер
#   2. g++ + наш шим: исполняемый файл, НЕ требующий libcudart/libcuda
#   3. nvcc (link)  : ТОТ ЖЕ исходник против настоящего cudart — эталон
#   4. оба запускаются, выводы сравниваются; расхождение валит прогон
#
# GPU не нужен ни на одном шаге. Без nvcc тест честно сообщает SKIP.
#
# Required: -DNVCC -DPROGRAMS (;-список .cu) -DWORK_DIR -DSHIM -DTEST_BIN

foreach(req NVCC PROGRAMS WORK_DIR SHIM TEST_BIN)
  if(NOT DEFINED ${req} OR "${${req}}" STREQUAL "")
    message(STATUS "foreign program: SKIP (missing -D${req}=...)")
    return()
  endif()
endforeach()

foreach(req NVCC SHIM TEST_BIN)
  if(NOT EXISTS "${${req}}")
    message(STATUS "foreign program: SKIP (${req} not found: ${${req}})")
    return()
  endif()
endforeach()

file(MAKE_DIRECTORY "${WORK_DIR}")
find_program(GXX NAMES g++ clang++ REQUIRED)
string(REPLACE ";" ";" PROGRAM_LIST "${PROGRAMS}")

set(ANY_RAN FALSE)
foreach(src ${PROGRAM_LIST})
  get_filename_component(name "${src}" NAME_WE)
  set(obj "${WORK_DIR}/${name}.o")
  set(prog "${WORK_DIR}/${name}")
  set(prog_ref "${WORK_DIR}/${name}_ref")

  execute_process(COMMAND "${NVCC}" -arch=sm_75 -c "${src}" -o "${obj}"
    RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "foreign program: FAILED - ${name}: nvcc -c exited ${rc}: ${err}")
  endif()

  # The link names NO CUDA library on purpose: every CUDA symbol must resolve
  # from our own shim. --no-as-needed keeps the DT_NEEDED entry even though no
  # symbol is referenced from the library at link time, so the binary is loaded
  # with the shim at startup (same trick as a libcuda.so.1 imposter in
  # LD_LIBRARY_PATH).
  get_filename_component(SHIM_DIR "${SHIM}" DIRECTORY)
  get_filename_component(SHIM_NAME "${SHIM}" NAME)
  execute_process(
    COMMAND "${GXX}" -O2 "${obj}" -o "${prog}"
            -Wl,--no-as-needed -l:${SHIM_NAME} -L${SHIM_DIR} -Wl,-rpath,${SHIM_DIR}
    RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "foreign program: FAILED - ${name}: link exited ${rc}: ${err}")
  endif()

  execute_process(COMMAND "${NVCC}" -arch=sm_75 "${src}" -o "${prog_ref}"
    RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "foreign program: FAILED - ${name}: nvcc could not link the "
                        "reference binary (exit ${rc}): ${err}")
  endif()

  execute_process(
    COMMAND ${CMAKE_COMMAND} -E env "LD_LIBRARY_PATH=${SHIM_DIR}"
            "${TEST_BIN}" "${prog_ref}" "${prog}" "${SHIM}"
    RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
  set(ANY_RAN TRUE)
  string(REPLACE "\n" "\n    " pretty "${out}${err}")
  message(STATUS "foreign program [${name}]: ${pretty}")
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "foreign program: FAILED - ${name} did not run without NVIDIA "
                        "(exit ${rc})")
  endif()
endforeach()

if(NOT ANY_RAN)
  message(STATUS "foreign program: no programs ran")
endif()