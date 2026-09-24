# ---- Test 1: structural translation check (no LLVM toolchain needed) ----
add_executable(test_ptx_to_llvm test_ptx_to_llvm.cpp)
target_link_libraries(test_ptx_to_llvm PRIVATE rouge-compiler-lib rouge-ptx)
add_test(NAME compiler_ptx_to_llvm COMMAND test_ptx_to_llvm)

# ---- Test 1b: shared memory + bar.sync lowering (structural) ----
add_executable(test_ptx_to_llvm_shared test_ptx_to_llvm_shared.cpp)
target_link_libraries(test_ptx_to_llvm_shared PRIVATE rouge-compiler-lib rouge-ptx)
add_test(NAME compiler_ptx_to_llvm_shared COMMAND test_ptx_to_llvm_shared)

# ---- Test 1c: atomics + f16/bf16 lowering + exported .shared metadata ----
add_executable(test_ptx_to_llvm_atomics_fp16 test_ptx_to_llvm_atomics_fp16.cpp)
target_link_libraries(test_ptx_to_llvm_atomics_fp16 PRIVATE rouge-compiler-lib rouge-ptx)
add_test(NAME compiler_ptx_to_llvm_atomics_fp16 COMMAND test_ptx_to_llvm_atomics_fp16)

set(VADD_PTX ${CMAKE_CURRENT_SOURCE_DIR}/kernels/vadd.ptx)
set(BLOCK_REDUCE_PTX ${CMAKE_CURRENT_SOURCE_DIR}/kernels/block_reduce.ptx)
set(ATOMIC_REDUCE_PTX ${CMAKE_CURRENT_SOURCE_DIR}/kernels/atomic_reduce.ptx)
set(FP16_REDUCE_PTX ${CMAKE_CURRENT_SOURCE_DIR}/kernels/fp16_reduce.ptx)
set(GEMM_TILE_PTX ${CMAKE_CURRENT_SOURCE_DIR}/kernels/gemm_tile.ptx)

# Stage 1 always runs: PTX -> LLVM IR via our translator CLI.
foreach(kern vadd block_reduce atomic_reduce fp16_reduce gemm_tile)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${kern}.ll
    COMMAND ptx2ir ${CMAKE_CURRENT_SOURCE_DIR}/kernels/${kern}.ptx
            ${CMAKE_CURRENT_BINARY_DIR}/${kern}.ll
    DEPENDS ptx2ir ${CMAKE_CURRENT_SOURCE_DIR}/kernels/${kern}.ptx
    COMMENT "[rouge-compiler] ptx2ir: ${kern}.ptx -> ${kern}.ll")
endforeach()

# ---- Test 2: AOT native execution (needs clang to compile the IR) ----
find_program(CLANG_EXECUTABLE clang)
if(CLANG_EXECUTABLE)
  # Compile one IR module per canonical kernel: ptx2ir -> clang -O2 -> .o
  foreach(kern vadd block_reduce atomic_reduce fp16_reduce gemm_tile)
    add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${kern}.o
      COMMAND ${CLANG_EXECUTABLE} -O2 -c ${CMAKE_CURRENT_BINARY_DIR}/${kern}.ll
              -o ${CMAKE_CURRENT_BINARY_DIR}/${kern}.o
      DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${kern}.ll
      COMMENT "[rouge-compiler] clang: ${kern}.ll -> native object (AOT)")
  endforeach()

  add_executable(test_aot_native test_aot_native.cpp ${CMAKE_CURRENT_BINARY_DIR}/vadd.o)
  add_test(NAME aot_native_vadd COMMAND test_aot_native)

  # Test 2b: shared memory + bar.sync end-to-end (block_reduce kernel).
  add_executable(test_aot_block_reduce test_aot_block_reduce.cpp
                 ${CMAKE_CURRENT_BINARY_DIR}/block_reduce.o)
  target_link_libraries(test_aot_block_reduce PRIVATE rouge-runtime)
  add_test(NAME aot_native_block_reduce COMMAND test_aot_block_reduce)

  # Test 2c: relaxed atomics (atom.add/red.add as real hardware atomics).
  add_executable(test_aot_atomic_reduce test_aot_atomic_reduce.cpp
                 ${CMAKE_CURRENT_BINARY_DIR}/atomic_reduce.o)
  target_link_libraries(test_aot_atomic_reduce PRIVATE rouge-runtime)
  add_test(NAME aot_native_atomic_reduce COMMAND test_aot_atomic_reduce)

  # Test 2d: FP16/BF16 pipeline (half/bfloat, fma, f16 shared tile).
  add_executable(test_aot_fp16_reduce test_aot_fp16_reduce.cpp
                 ${CMAKE_CURRENT_BINARY_DIR}/fp16_reduce.o)
  target_link_libraries(test_aot_fp16_reduce PRIVATE rouge-runtime)
  add_test(NAME aot_native_fp16_reduce COMMAND test_aot_fp16_reduce)

  # ---- Test 3 (informational): same IR through the RISC-V backend ----
  # Compiles the IR for riscv64 with the vector extension (rv64gcv). Reports
  # status; skipped gracefully when this LLVM build lacks a riscv target.
  foreach(kern vadd atomic_reduce fp16_reduce)
    add_test(NAME compiler_rvv_backend_${kern}
      COMMAND ${CMAKE_COMMAND}
        -DCLANG=${CLANG_EXECUTABLE}
        -DIR_FILE=${CMAKE_CURRENT_BINARY_DIR}/${kern}.ll
        -DOUT_FILE=${CMAKE_CURRENT_BINARY_DIR}/${kern}-rv64.o
        -P ${CMAKE_CURRENT_SOURCE_DIR}/check_rvv_backend.cmake)
  endforeach()
endif()
