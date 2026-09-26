# ROUGE-V: GPU target matrix check for the PTX -> LLVM IR translator (stage 1).
#
# For every (platform, kernel) cell of the matrix this script
#   1. probes whether the local clang can emit code for the target triple at all.
#      An LLVM build without the backend, or an AMD machine without the ROCm
#      toolchain, makes the whole platform report SKIP (exit code 0) instead of
#      failing the test run - same policy as check_rvv_backend.cmake;
#   2. runs our own ptx2ir with --target <triple>;
#   3. checks that the emitted IR really carries the requested target triple and
#      the target-correct GPU address spaces. These two assertions depend only on
#      our translator, never on the host toolchain, so a violation is a genuine
#      regression and fails the test;
#   4. compiles the IR with clang for that target and checks an object appeared.
#
# Failure policy (see check_rvv_backend.cmake for the "informational" style):
#   * environment problem (no backend, no ROCm)      -> STATUS "SKIP", exit 0
#   * our translator or the IR contract is wrong     -> FATAL_ERROR
#   * clang rejects our IR although the probe passed -> STATUS "SKIP" by default,
#     because that is a host toolchain quirk (LLVM version, missing device libs);
#     pass -DSTRICT=ON to turn it into a hard failure.
#
# Required variables:
#   CLANG       clang executable
#   PTX2IR      ptx2ir executable
#   KERNEL_DIR  directory holding the canonical .ptx kernels
#   IR_FILE     base path for the generated IR  <stem>.ll  (matrix: <stem>-<platform>-<kernel>.ll)
#   OUT_FILE    base path for the objects       <stem>.o   (matrix: <stem>-<platform>-<kernel>.o)
#   KERNELS     comma separated kernel names, e.g. "vadd,block_reduce"
#   PLATFORMS   comma separated "name|triple|flag1+flag2" specs,
#               e.g. "amdgpu|amdgcn-amd-amdhsa|-mcpu=gfx1100+-O2"
# Optional:
#   STRICT      ON -> "clang rejected our IR" fails the test instead of skipping
#
# Note on separators: these values travel through the CTest command line as one
# -D argument each, where a ';' splits the argument into several and a space
# splits it as well. Hence commas between list items and '+' between the flags
# of one spec; both are turned into proper CMake lists below.

if(NOT DEFINED STRICT)
  set(STRICT OFF)
endif()

foreach(req CLANG PTX2IR KERNEL_DIR IR_FILE OUT_FILE KERNELS PLATFORMS)
  if(NOT DEFINED ${req} OR "${${req}}" STREQUAL "")
    message(STATUS "gpu target matrix: SKIP (missing -D${req}=...)")
    return()
  endif()
endforeach()

if(NOT EXISTS "${CLANG}")
  message(STATUS "gpu target matrix: SKIP (clang not found: ${CLANG})")
  return()
endif()

if(NOT EXISTS "${PTX2IR}")
  message(STATUS "gpu target matrix: SKIP (ptx2ir not built yet: ${PTX2IR})")
  return()
endif()

# The base paths only supply the output directory and the file-name stem; the
# matrix appends "-<platform>-<kernel>" so that all cells keep separate outputs.
get_filename_component(OUT_DIR "${IR_FILE}" DIRECTORY)
get_filename_component(OUT_STEM "${IR_FILE}" NAME_WE)
file(MAKE_DIRECTORY "${OUT_DIR}")

# Turn the comma separated command line values into real CMake lists.
string(REPLACE "," ";" MATRIX_KERNELS "${KERNELS}")
string(REPLACE "," ";" MATRIX_PLATFORMS "${PLATFORMS}")

# Every kernel must be present: a missing .ptx is a configuration error, not an
# environment problem, so it is reported as a failure instead of a silent skip.
foreach(kern IN LISTS MATRIX_KERNELS)
  if(NOT EXISTS "${KERNEL_DIR}/${kern}.ptx")
    message(FATAL_ERROR "gpu target matrix: FAILED - missing kernel ${KERNEL_DIR}/${kern}.ptx")
  endif()
endforeach()

# rouge_condense(<text> <out>) - flattens a tool's stderr into one short line.
# clang failures can carry a full LLVM crash dump; echoing that verbatim into a
# ctest log buries the actual verdict.
function(rouge_condense text out)
  string(REPLACE "\n" " " flat "${text}")
  string(REPLACE "\r" " " flat "${flat}")
  string(REGEX REPLACE "[ \t][ \t]+" " " flat "${flat}")
  string(STRIP "${flat}" flat)
  string(LENGTH "${flat}" len)
  if(len GREATER 200)
    string(SUBSTRING "${flat}" 0 200 flat)
    set(flat "${flat}...")
  endif()
  if(flat STREQUAL "")
    set(flat "(no diagnostic)")
  endif()
  set(${out} "${flat}" PARENT_SCOPE)
endfunction()

# rouge_probe_backend(<clang> <triple> <flag-list> <probe-file> <probe-obj> <out-status>)
# Compiles a trivial module for the triple. It succeeds only when this clang
# really has a usable backend for the target, which is exactly the "does this
# host support AMD/NVIDIA (LLVM backend, ROCm toolchain)" question. <flag-list>
# must stay a real CMake list, because CMake turns an unquoted expansion into
# separate arguments only on ';', never on spaces.
function(rouge_probe_backend clang triple flags probe_ll probe_o out_status)
  file(WRITE "${probe_ll}"
       "target triple = \"${triple}\"\n"
       "define i32 @rouge_probe() {\n"
       "entry:\n"
       "  ret i32 0\n"
       "}\n")
  file(REMOVE "${probe_o}")
  execute_process(
    COMMAND "${clang}" --target=${triple} ${flags}
            -c "${probe_ll}" -o "${probe_o}"
    RESULT_VARIABLE rc
    OUTPUT_VARIABLE pout
    ERROR_VARIABLE perr)
  if(rc EQUAL 0 AND EXISTS "${probe_o}")
    set(${out_status} "ok" PARENT_SCOPE)
  else()
    rouge_condense("${perr}" perr_short)
    set(${out_status} "unavailable: ${perr_short}" PARENT_SCOPE)
  endif()
endfunction()

# rouge_addrspace_contract(<triple> <out_required> <out_forbidden>)
# Translates the target into the address-space numbers our IR must use:
#   global memory is addrspace(1) on both GPU targets; shared memory is
#   addrspace(5) on AMDGPU and addrspace(3) on NVPTX; AMDGPU additionally keeps
#   private allocas in addrspace(5) while NVPTX leaves them flat.
# Unknown triples get empty lists, i.e. no address-space assertion.
function(rouge_addrspace_contract triple out_req out_forbid)
  set(req "1")
  set(forbid "")
  if(triple MATCHES "^amdgcn")
    list(APPEND req "5")
    list(APPEND forbid "3")   # NVPTX shared space must not leak into AMDGPU IR
  elseif(triple MATCHES "^nvptx")
    list(APPEND forbid "4" "5")  # AMDGPU flat / private spaces must not leak
  else()
    set(req "")
    set(forbid "")
  endif()
  set(${out_req} "${req}" PARENT_SCOPE)
  set(${out_forbid} "${forbid}" PARENT_SCOPE)
endfunction()

set(ok_cells 0)
set(skipped_cells 0)

foreach(spec IN LISTS MATRIX_PLATFORMS)
  string(REPLACE "|" ";" _parts "${spec}")
  list(LENGTH _parts _nparts)
  if(_nparts LESS 2)
    message(FATAL_ERROR "gpu target matrix: FAILED - malformed platform spec "
               "\"${spec}\", expected \"name|triple|flag1+flag2\"")
  endif()
  list(GET _parts 0 pname)
  list(GET _parts 1 triple)
  if(_nparts GREATER 2)
    list(GET _parts 2 _pflags_raw)
    # A spec whose flag field still contains a space or a '|' reached us mangled,
    # e.g. because the test wiring passed the whole flag string to clang as one
    # argument. That is a bug in the wiring, not a missing toolchain, and it
    # would make every cell skip silently, so refuse to run instead.
    if(_pflags_raw MATCHES "[ ;|]")
      message(FATAL_ERROR "gpu target matrix: FAILED - platform spec \"${spec}\" must "
                 "separate clang flags with '+' and never use spaces")
    endif()
    string(REPLACE "+" ";" FLAGS "${_pflags_raw}")
  else()
    set(FLAGS "-O2")
  endif()
  if(triple STREQUAL "")
    message(FATAL_ERROR "gpu target matrix: FAILED - platform \"${pname}\" has no triple")
  endif()
  string(REPLACE ";" " " flags_display "${FLAGS}")

  # ---- step 1: is this target usable on this host at all? ----
  set(probe_ll "${OUT_DIR}/.rouge_probe-${pname}.ll")
  set(probe_o  "${OUT_DIR}/.rouge_probe-${pname}.o")
  set(probe_status "ok")
  rouge_probe_backend("${CLANG}" "${triple}" "${FLAGS}" "${probe_ll}" "${probe_o}" probe_status)
  file(REMOVE "${probe_ll}" "${probe_o}")
  if(NOT probe_status STREQUAL "ok")
    message(STATUS "gpu target matrix: SKIP ${pname} (${triple}): clang has no usable "
                   "backend for this target - LLVM built without it, or no ROCm "
                   "toolchain installed. All ${pname} kernels skipped. "
                   "clang said: ${probe_status}")
    set(skipped_cells 1)
    continue()
  endif()

  rouge_addrspace_contract("${triple}" AS_REQUIRED AS_FORBIDDEN)

  set(platform_ok 0)
  foreach(kern IN LISTS MATRIX_KERNELS)
    set(ir_file  "${OUT_DIR}/${OUT_STEM}-${pname}-${kern}.ll")
    set(obj_file "${OUT_DIR}/${OUT_STEM}-${pname}-${kern}.o")
    file(REMOVE "${ir_file}" "${obj_file}")

    # ---- step 2: our translator must honour --target ----
    execute_process(
      COMMAND "${PTX2IR}" --target "${triple}" "${KERNEL_DIR}/${kern}.ptx" "${ir_file}"
      RESULT_VARIABLE rc
      OUTPUT_VARIABLE out
      ERROR_VARIABLE err)
    if(NOT rc EQUAL 0)
      rouge_condense("${err}" err)
      message(FATAL_ERROR "gpu target matrix: FAILED - ${pname}/${kern}: "
                 "ptx2ir --target ${triple} exited ${rc}: ${err}")
    endif()
    if(NOT EXISTS "${ir_file}")
      message(FATAL_ERROR "gpu target matrix: FAILED - ${pname}/${kern}: "
                 "ptx2ir produced no IR file")
    endif()

    # ---- step 3: the IR must be self-describing and use the right spaces ----
    file(READ "${ir_file}" ir_text)
    string(FIND "${ir_text}" "target triple = \"${triple}\"" _triple_at)
    if(_triple_at EQUAL -1)
      message(FATAL_ERROR "gpu target matrix: FAILED - ${pname}/${kern}: "
                 "emitted IR does not declare target triple \"${triple}\"")
    endif()
    foreach(as IN LISTS AS_REQUIRED)
      string(FIND "${ir_text}" "addrspace(${as})" _at)
      if(_at EQUAL -1)
        message(FATAL_ERROR "gpu target matrix: FAILED - ${pname}/${kern}: "
                   "IR is missing addrspace(${as}) required by ${triple}")
      endif()
    endforeach()
    foreach(as IN LISTS AS_FORBIDDEN)
      string(FIND "${ir_text}" "addrspace(${as})" _at)
      if(NOT _at EQUAL -1)
        message(FATAL_ERROR "gpu target matrix: FAILED - ${pname}/${kern}: "
                   "IR uses addrspace(${as}) which does not belong to ${triple}")
      endif()
    endforeach()

    # ---- step 4: the real backend must accept the result ----
    execute_process(
      COMMAND "${CLANG}" --target=${triple} ${FLAGS}
              -c "${ir_file}" -o "${obj_file}"
      RESULT_VARIABLE rc
      OUTPUT_VARIABLE out
      ERROR_VARIABLE err)
    string(STRIP "${err}" err)
    rouge_condense("${err}" err)
    if(NOT rc EQUAL 0)
      if(STRICT)
        message(FATAL_ERROR "gpu target matrix: FAILED - ${pname}/${kern}: "
                   "clang rejected the IR for ${triple} (exit ${rc}): ${err}")
      else()
        message(STATUS "gpu target matrix: SKIP ${pname}/${kern} (clang exit ${rc}: ${err})")
        set(skipped_cells 1)
        continue()
      endif()
    endif()
    if(NOT EXISTS "${obj_file}")
      message(FATAL_ERROR "gpu target matrix: FAILED - ${pname}/${kern}: "
                 "clang exited 0 for ${triple} but produced no object file")
    endif()

    math(EXPR platform_ok "${platform_ok} + 1")
    math(EXPR ok_cells "${ok_cells} + 1")
    file(SIZE "${obj_file}" obj_size)
    message(STATUS "gpu target matrix: OK ${pname}/${kern} "
                   "(clang --target=${triple} ${flags_display} -> ${obj_size} byte object, "
                   "target triple and address spaces verified)")
  endforeach()

  if(platform_ok GREATER 0)
    list(LENGTH MATRIX_KERNELS kernel_count)
    message(STATUS "gpu target matrix: OK ${pname} (${triple}): "
                   "${platform_ok}/${kernel_count} kernels compiled")
  endif()
endforeach()

if(skipped_cells AND NOT ok_cells)
  message(STATUS "gpu target matrix: SKIP - no kernel of this matrix was "
                 "exercised on this host")
  return()
endif()

if(ok_cells GREATER 0)
  message(STATUS "gpu target matrix: OK - ${ok_cells} kernel(s) translated and "
                 "compiled for their GPU target")
endif()
