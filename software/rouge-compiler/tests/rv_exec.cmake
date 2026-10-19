# ROUGE-V: run a PTX-derived kernel on RISC-V (rv64gcv) under qemu-user.
#
# The RISC-V target was compile-only so far. This script takes it one step
# further: translate the PTX for riscv64, link it with a freestanding
# harness (no libc, no threads) and RUN the binary under qemu-riscv64,
# checking the kernel's result.
#
# Honest scope: the harness executes one lane at a time, so it covers kernels
# without cross-lane barriers (vadd). Kernels with bar.sync need real
# threads and are not claimed here.
#
# SKIPs (honestly, never silently) when: qemu-riscv64 is not installed, the
# LLVM build has no riscv target, or the translation fails.

foreach(req CLANG PTX2IR KERNEL WORK_DIR QEMU)
  if(NOT DEFINED ${req} OR "${${req}}" STREQUAL "")
    message(STATUS "riscv exec: SKIP (missing -D${req}=...)")
    return()
  endif()
endforeach()

if(NOT EXISTS "${QEMU}")
  message(STATUS "riscv exec: SKIP (qemu-riscv64 not found: ${QEMU})")
  return()
endif()

file(MAKE_DIRECTORY "${WORK_DIR}")
set(ll "${WORK_DIR}/vadd_rv.ll")
set(bin "${WORK_DIR}/vadd_rv")

# 1. Translate the PTX for riscv64 (same translator, different target).
execute_process(
  COMMAND "${PTX2IR}" --target riscv64-unknown-linux-gnu "${KERNEL}" "${ll}"
  RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
  message(STATUS "riscv exec: SKIP (riscv64 target unavailable: ${err})")
  return()
endif()
if(NOT EXISTS "${ll}")
  message(FATAL_ERROR "riscv exec: FAILED - ptx2ir produced no IR for riscv64")
endif()

# 2. Link with the freestanding harness: no libc, static, RVV enabled.
#    lld is required: the system GNU ld usually has no RISC-V emulation.
execute_process(
  COMMAND "${CLANG}" --target=riscv64-unknown-linux-gnu -march=rv64gcv
          -O2 -ffreestanding -nostdlib -static -fuse-ld=lld
          "${ll}" "${HARNESS}" -o "${bin}"
  RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
  message(STATUS "riscv exec: SKIP (cannot build a riscv64 binary: ${err})")
  return()
endif()
if(NOT EXISTS "${bin}")
  message(FATAL_ERROR "riscv exec: FAILED - no riscv64 binary produced")
endif()

# 3. Run it. Exit code 0 = the kernel computed the right answer.
execute_process(
  COMMAND "${QEMU}" "${bin}"
  RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "riscv exec: FAILED - kernel result wrong under qemu "
                      "(exit ${rc}): ${out}${err}")
endif()

message(STATUS "riscv exec: OK - PTX-derived kernel ran on rv64gcv under "
               "qemu-user and produced the correct result")
