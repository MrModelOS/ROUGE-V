# ROUGE-V: Tier-0 foreign-PTX subset sweep (no GPU, no nvcc needed).
#
# Pushes the small foreign-feature probes in KERNEL_DIR/foreign/*.ptx through
#   ptx2ir (host triple) -> clang -c (host object)
#   ptx2ir --target nvptx64-nvidia-cuda -> clang --target=nvptx64 (NVPTX object)
# Anything else (translate error, backend rejection) is OUR bug -> FATAL_ERROR,
# with one designed exception: probe_clock MUST fail on the host triple
# (%clock64 has no host equivalent) and MUST pass on NVPTX (real sreg).
#
# Required variables:
#   CLANG       clang executable
#   PTX2IR      ptx2ir executable
#   KERNEL_DIR  directory holding kernels/ (probes live in kernels/foreign/)
#   WORK_DIR    scratch directory for the generated .ll / .o files
#   SM          target architecture for the NVPTX clang step, e.g. "sm_75"

foreach(req CLANG PTX2IR KERNEL_DIR WORK_DIR SM)
  if(NOT DEFINED ${req} OR "${${req}}" STREQUAL "")
    message(STATUS "foreign ptx sweep: SKIP (missing -D${req}=...)")
    return()
  endif()
endforeach()

foreach(req CLANG PTX2IR)
  if(NOT EXISTS "${${req}}")
    message(STATUS "foreign ptx sweep: SKIP (${req} not found: ${${req}})")
    return()
  endif()
endforeach()

file(MAKE_DIRECTORY "${WORK_DIR}")

function(rouge_probe_cell name)
  set(ptx "${KERNEL_DIR}/foreign/${name}.ptx")
  if(NOT EXISTS "${ptx}")
    message(FATAL_ERROR "foreign ptx sweep: FAILED - missing probe ${ptx}")
  endif()

  foreach(target host nvptx)
    set(ll "${WORK_DIR}/${name}.${target}.ll")
    set(obj "${WORK_DIR}/${name}.${target}.o")
    if(target STREQUAL "host")
      set(cmd_ptx2ir "${PTX2IR}" "${ptx}" "${ll}")
      set(cmd_cc "${CLANG}" -O2 -c "${ll}" -o "${obj}")
    else()
      set(cmd_ptx2ir "${PTX2IR}" --target nvptx64-nvidia-cuda "${ptx}" "${ll}")
      set(cmd_cc "${CLANG}" --target=nvptx64-nvidia-cuda -march=${SM} -O2
                 -c "${ll}" -o "${obj}")
    endif()

    execute_process(COMMAND ${cmd_ptx2ir}
      RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
    set(expect_fail FALSE)
    if(name STREQUAL "probe_clock" AND target STREQUAL "host")
      set(expect_fail TRUE)
    endif()
    if(expect_fail)
      if(rc EQUAL 0)
        message(FATAL_ERROR "foreign ptx sweep: FAILED - ${name}.${target}: "
                            "expected a loud host refusal for %clock64, got rc=0")
      endif()
      if(NOT "${err}" MATCHES "clock64")
        message(FATAL_ERROR "foreign ptx sweep: FAILED - ${name}.${target}: "
                            "refusal does not name %clock64: ${err}")
      endif()
      message(STATUS "foreign ptx sweep: OK ${name}.${target} "
                     "(loud host refusal as designed)")
      continue()
    endif()
    if(NOT rc EQUAL 0)
      message(FATAL_ERROR "foreign ptx sweep: FAILED - ${name}.${target}: "
                          "ptx2ir exited ${rc}: ${err}")
    endif()

    execute_process(COMMAND ${cmd_cc}
      RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
    if(NOT rc EQUAL 0)
      message(FATAL_ERROR "foreign ptx sweep: FAILED - ${name}.${target}: "
                          "clang rejected the IR (exit ${rc}): ${err}")
    endif()
    if(NOT EXISTS "${obj}")
      message(FATAL_ERROR "foreign ptx sweep: FAILED - ${name}.${target}: "
                          "clang produced no object")
    endif()
    message(STATUS "foreign ptx sweep: OK ${name}.${target} "
                   "(translate + assemble)")
  endforeach()
endfunction()

rouge_probe_cell(probe_barrier)
rouge_probe_cell(probe_volatile)
rouge_probe_cell(probe_local)
rouge_probe_cell(probe_dynshared)
rouge_probe_cell(probe_clock)

message(STATUS "foreign ptx sweep: OK - all probe cells verified")
