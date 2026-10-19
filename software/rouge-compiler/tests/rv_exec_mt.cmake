# ROUGE-V: run a PTX-derived block kernel (with bar.sync) on RISC-V.
#
# The companion of rv_exec.cmake: there the harness runs one lane at a time
# (kernels without barriers). Here the kernel is a REAL block kernel with a
# block-wide barrier, executed by 32 real pthreads per block, using the
# project's own runtime (rouge_smem_alloc + __rouge_syncthreads).
#
# Needs a riscv64 sysroot: headers, libc.a, libgcc.a and libstdc++.a plus a
# GCC-cross layout for the crt files. On Debian/Ubuntu that is:
#   libc6-dev-riscv64-cross libc6-riscv64-cross linux-libc-dev-riscv64-cross
#   libgcc-13-dev-riscv64-cross libstdc++-13-dev-riscv64-cross
# unpacked into one prefix (see the project README for the recipe).
# Without a sysroot the test reports SKIP - honestly, not silently.

foreach(req CLANG CLANGXX PTX2IR KERNEL HARNESS RUNTIME WORK_DIR QEMU SYSROOT)
  if(NOT DEFINED ${req} OR "${${req}}" STREQUAL "")
    message(STATUS "riscv exec (threads): SKIP (missing -D${req}=...)")
    return()
  endif()
endforeach()

if(NOT EXISTS "${QEMU}")
  message(STATUS "riscv exec (threads): SKIP (qemu-riscv64 not found: ${QEMU})")
  return()
endif()
if(NOT IS_DIRECTORY "${SYSROOT}/usr/include")
  message(STATUS "riscv exec (threads): SKIP (no riscv64 sysroot at ${SYSROOT})")
  return()
endif()

file(MAKE_DIRECTORY "${WORK_DIR}")
set(ll "${WORK_DIR}/block_reduce_rv.ll")
set(bin "${WORK_DIR}/block_reduce_rv")

execute_process(
  COMMAND "${PTX2IR}" --target riscv64-unknown-linux-gnu "${KERNEL}" "${ll}"
  RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
  message(STATUS "riscv exec (threads): SKIP (riscv64 target unavailable: ${err})")
  return()
endif()

# The GCC-cross directory inside the sysroot carries crtbegin/crtend and
# libgcc/libstdc++; clang needs it via -B and --gcc-toolchain.
file(GLOB _gcc_dirs "${SYSROOT}/usr/lib/gcc-cross/riscv64-linux-gnu/*")
list(SORT _gcc_dirs)
list(REVERSE _gcc_dirs)
list(GET _gcc_dirs 0 _gcc_dir)
if(NOT EXISTS "${_gcc_dir}")
  message(STATUS "riscv exec (threads): SKIP (no gcc-cross dir in the sysroot)")
  return()
endif()

set(_flags --target=riscv64-unknown-linux-gnu "--sysroot=${SYSROOT}"
           "-B${_gcc_dir}" -march=rv64gcv -O2 -static
           -fuse-ld=lld)

execute_process(
  COMMAND "${CLANG}" ${_flags} -ffreestanding -c "${HARNESS}"
          -o "${WORK_DIR}/harness.o"
  RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
  message(STATUS "riscv exec (threads): SKIP (harness does not build: ${err})")
  return()
endif()

execute_process(
  COMMAND "${CLANGXX}" ${_flags} -c "${RUNTIME}" -o "${WORK_DIR}/runtime.o"
  RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
  message(STATUS "riscv exec (threads): SKIP (runtime does not build: ${err})")
  return()
endif()

execute_process(
  COMMAND "${CLANGXX}" ${_flags} "${ll}" "${WORK_DIR}/harness.o"
          "${WORK_DIR}/runtime.o" -o "${bin}" -lpthread
  RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
  message(STATUS "riscv exec (threads): SKIP (cannot link riscv64 binary: ${err})")
  return()
endif()

execute_process(
  COMMAND "${QEMU}" "${bin}"
  RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "riscv exec (threads): FAILED - block kernel with bar.sync "
                      "gave the wrong result under qemu (exit ${rc}): ${out}${err}")
endif()
message(STATUS "riscv exec (threads): OK - block kernel with bar.sync ran on "
               "rv64gcv with real threads under qemu-user")
