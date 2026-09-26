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

# ---- Test 1d: cas / exch / min / max lowering ----
add_executable(test_ptx_to_llvm_atom_cas test_ptx_to_llvm_atom_cas.cpp)
target_link_libraries(test_ptx_to_llvm_atom_cas PRIVATE rouge-compiler-lib rouge-ptx)
add_test(NAME compiler_ptx_to_llvm_atom_cas COMMAND test_ptx_to_llvm_atom_cas)

set(VADD_PTX ${CMAKE_CURRENT_SOURCE_DIR}/kernels/vadd.ptx)
set(BLOCK_REDUCE_PTX ${CMAKE_CURRENT_SOURCE_DIR}/kernels/block_reduce.ptx)
set(ATOMIC_REDUCE_PTX ${CMAKE_CURRENT_SOURCE_DIR}/kernels/atomic_reduce.ptx)
set(FP16_REDUCE_PTX ${CMAKE_CURRENT_SOURCE_DIR}/kernels/fp16_reduce.ptx)
set(GEMM_TILE_PTX ${CMAKE_CURRENT_SOURCE_DIR}/kernels/gemm_tile.ptx)
set(ATOM_CAS_PTX ${CMAKE_CURRENT_SOURCE_DIR}/kernels/atom_cas.ptx)
set(SHFL_REDUCE_PTX ${CMAKE_CURRENT_SOURCE_DIR}/kernels/shfl_reduce.ptx)

# ---- Test 1e: warp shuffles + vote + activemask (structural) ----
add_executable(test_ptx_to_llvm_warp test_ptx_to_llvm_warp.cpp)
target_link_libraries(test_ptx_to_llvm_warp PRIVATE rouge-compiler-lib rouge-ptx)
add_test(NAME compiler_ptx_to_llvm_warp COMMAND test_ptx_to_llvm_warp)

# Stage 1 always runs: PTX -> LLVM IR via our translator CLI.
foreach(kern vadd block_reduce atomic_reduce fp16_reduce gemm_tile atom_cas shfl_reduce)
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
  foreach(kern vadd block_reduce atomic_reduce fp16_reduce gemm_tile atom_cas shfl_reduce)
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

  # Test 2e: tiled GEMM 16x16 in F16 via shared memory (gemm_tile kernel).
  add_executable(test_aot_gemm_tile test_aot_gemm_tile.cpp
                 ${CMAKE_CURRENT_BINARY_DIR}/gemm_tile.o)
  target_link_libraries(test_aot_gemm_tile PRIVATE rouge-runtime)
  add_test(NAME aot_native_gemm_tile COMMAND test_aot_gemm_tile)

  # Test 2f: cas / exch / min / max (atom_cas kernel).
  add_executable(test_aot_atom_cas test_aot_atom_cas.cpp
                 ${CMAKE_CURRENT_BINARY_DIR}/atom_cas.o)
  target_link_libraries(test_aot_atom_cas PRIVATE rouge-runtime)
  add_test(NAME aot_native_atom_cas COMMAND test_aot_atom_cas)

  # Test 2g: warp shuffle reduction (shfl.bfly — one warp per block, AOT scalar fallback).
  add_executable(test_aot_shfl_reduce test_aot_shfl_reduce.cpp
                 ${CMAKE_CURRENT_BINARY_DIR}/shfl_reduce.o)
  target_link_libraries(test_aot_shfl_reduce PRIVATE rouge-runtime)
  add_test(NAME aot_native_shfl_reduce COMMAND test_aot_shfl_reduce)

  # ---- Test 4: real NVIDIA PTX (vendor input, not our own .ptx files) ----
  # Compiles real CUDA C with nvcc and pushes the vendor PTX through the whole
  # ptx2ir -> clang chain. Skips cleanly when nvcc is absent.
  find_program(NVCC_EXECUTABLE nvcc)
  if(NVCC_EXECUTABLE)
    add_executable(test_nvcc_ptx test_nvcc_ptx.cpp)
    target_compile_definitions(test_nvcc_ptx PRIVATE
      "RougeTestDir=\"${CMAKE_CURRENT_SOURCE_DIR}\""
      "RougePtx2Ir=\"$<TARGET_FILE:ptx2ir>\"")
    add_test(NAME compiler_nvcc_ptx COMMAND test_nvcc_ptx ${NVCC_EXECUTABLE}
             ${CLANG_EXECUTABLE})
    set_tests_properties(compiler_nvcc_ptx PROPERTIES SKIP_RETURN_CODE 77)
  else()
    message(STATUS "rouge-compiler: nvcc not found — vendor-PTX test disabled")
  endif()

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

  # ---- Tests 5 + 6 (informational): GPU target matrix ----
  # ptx2ir --target <triple> must emit IR that carries the requested target
  # triple plus the target-correct GPU address spaces, and that IR must survive
  # a real LLVM GPU backend. The matrix script probes each backend first, so on
  # a host whose LLVM lacks it (or, for AMD, without the ROCm toolchain) the
  # platform is reported as SKIP and never fails the run.

  # Kernels exercised by the GPU matrix. vadd/block_reduce/atomic_reduce are the
  # plain kernels, fp16_reduce and gemm_tile additionally use the shared-memory
  # address space, so the matrix covers both kinds of address-space mapping.
  set(GPU_MATRIX_KERNELS vadd block_reduce atomic_reduce fp16_reduce gemm_tile)

  # GPU target used by the AMD matrix; override on the command line to target a
  # different gfx architecture, e.g.
  #   -DROUGE_AMDGPU_TRIPLE=amdgcn-amd-amdhsa -DROUGE_AMDGPU_FLAGS="-mcpu=gfx90a -O2"
  set(ROUGE_AMDGPU_TRIPLE "amdgcn-amd-amdhsa" CACHE STRING
      "LLVM triple passed to ptx2ir --target for the AMD GPU matrix test")
  set(ROUGE_AMDGPU_FLAGS "-mcpu=gfx1100 -O2" CACHE STRING
      "Extra clang flags for the AMD GPU matrix test")

  # GPU target used by the NVIDIA matrix.
  set(ROUGE_NVPTX_TRIPLE "nvptx64-nvidia-cuda" CACHE STRING
      "LLVM triple passed to ptx2ir --target for the NVIDIA GPU matrix test")
  set(ROUGE_NVPTX_FLAGS "-march=sm_75 -O2" CACHE STRING
      "Extra clang flags for the NVIDIA GPU matrix test")

  # These values are handed to the matrix script as single -D arguments, where a
  # ';' or a space would be split by the CTest command line. The script therefore
  # expects ',' between list items and '+' between the clang flags of one spec.
  string(REPLACE " " "+" AMDGPU_FLAGS_SPEC "${ROUGE_AMDGPU_FLAGS}")
  string(REPLACE " " "+" NVPTX_FLAGS_SPEC "${ROUGE_NVPTX_FLAGS}")
  string(REPLACE ";" "," GPU_MATRIX_KERNELS_SPEC "${GPU_MATRIX_KERNELS}")

  add_test(NAME compiler_amdgpu_backend
    COMMAND ${CMAKE_COMMAND}
      -DCLANG=${CLANG_EXECUTABLE}
      -DPTX2IR=$<TARGET_FILE:ptx2ir>
      -DKERNEL_DIR=${CMAKE_CURRENT_SOURCE_DIR}/kernels
      -DIR_FILE=${CMAKE_CURRENT_BINARY_DIR}/vadd.ll
      -DOUT_FILE=${CMAKE_CURRENT_BINARY_DIR}/vadd-amdgpu.o
      -DKERNELS=${GPU_MATRIX_KERNELS_SPEC}
      -DPLATFORMS=amdgpu|${ROUGE_AMDGPU_TRIPLE}|${AMDGPU_FLAGS_SPEC}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/gpu_targets.cmake)

  add_test(NAME compiler_nvptx_backend
    COMMAND ${CMAKE_COMMAND}
      -DCLANG=${CLANG_EXECUTABLE}
      -DPTX2IR=$<TARGET_FILE:ptx2ir>
      -DKERNEL_DIR=${CMAKE_CURRENT_SOURCE_DIR}/kernels
      -DIR_FILE=${CMAKE_CURRENT_BINARY_DIR}/vadd.ll
      -DOUT_FILE=${CMAKE_CURRENT_BINARY_DIR}/vadd-nvptx.o
      -DKERNELS=${GPU_MATRIX_KERNELS_SPEC}
      -DPLATFORMS=nvptx|${ROUGE_NVPTX_TRIPLE}|${NVPTX_FLAGS_SPEC}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/gpu_targets.cmake)
endif()
