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
foreach(kern vadd block_reduce atomic_reduce fp16_reduce gemm_tile atom_cas shfl_reduce shfl_pred vote_pred shfl_pred_all fp8_cvt prmt_dp4a bitfield cmp_signed mma_tile mma_pipe)
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
  foreach(kern vadd block_reduce atomic_reduce fp16_reduce gemm_tile atom_cas shfl_reduce shfl_pred vote_pred shfl_pred_all fp8_cvt prmt_dp4a bitfield cmp_signed mma_tile mma_pipe)
    add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${kern}.o
      COMMAND ${CLANG_EXECUTABLE} -O2 -c ${CMAKE_CURRENT_BINARY_DIR}/${kern}.ll
              -o ${CMAKE_CURRENT_BINARY_DIR}/${kern}.o
      DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${kern}.ll
      COMMENT "[rouge-compiler] clang: ${kern}.ll -> native object (AOT)")
  endforeach()

  # The ldmatrix kernel has three entries, so ptx2ir runs once per entry (it
  # translates one .entry per module).
  foreach(entry ldmatrix_tile ldmatrix_trans ldmatrix_x4 ldmatrix_sm75 stmatrix_sm90)
    add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${entry}.ll
      COMMAND ptx2ir --entry ${entry}
              ${CMAKE_CURRENT_SOURCE_DIR}/kernels/ldmatrix_tile.ptx
              ${CMAKE_CURRENT_BINARY_DIR}/${entry}.ll
      DEPENDS ptx2ir ${CMAKE_CURRENT_SOURCE_DIR}/kernels/ldmatrix_tile.ptx
      COMMENT "[rouge-compiler] ptx2ir: ldmatrix_tile.ptx[${entry}] -> ${entry}.ll")
    add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${entry}.o
      COMMAND ${CLANG_EXECUTABLE} -O2 -c ${CMAKE_CURRENT_BINARY_DIR}/${entry}.ll
              -o ${CMAKE_CURRENT_BINARY_DIR}/${entry}.o
      DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${entry}.ll
      COMMENT "[rouge-compiler] clang: ${entry}.ll -> native object (AOT)")
  endforeach()

  # The mma forms kernel has sixteen entries (eight forms x ones/lane), so
  # ptx2ir runs once per entry (it translates one .entry per module).
  set(MMA_FORMS_OBJS "")
  foreach(entry
      mma_bf16k16f32_ones mma_bf16k16f32_lane
      mma_e4m3k16f32_ones mma_e4m3k16f32_lane
      mma_e5m2k16f32_ones mma_e5m2k16f32_lane
      mma_u8k16s32_ones mma_u8k16s32_lane
      mma_s8k16s32_ones mma_s8k16s32_lane
      mma_f16k16f16_ones mma_f16k16f16_lane
      mma_f16k8f32_ones mma_f16k8f32_lane
      mma_bf16k8f16_ones mma_bf16k8f16_lane)
    add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${entry}.ll
      COMMAND ptx2ir --entry ${entry}
              ${CMAKE_CURRENT_SOURCE_DIR}/kernels/mma_forms.ptx
              ${CMAKE_CURRENT_BINARY_DIR}/${entry}.ll
      DEPENDS ptx2ir ${CMAKE_CURRENT_SOURCE_DIR}/kernels/mma_forms.ptx
      COMMENT "[rouge-compiler] ptx2ir: mma_forms.ptx[${entry}] -> ${entry}.ll")
    add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${entry}.o
      COMMAND ${CLANG_EXECUTABLE} -O2 -c ${CMAKE_CURRENT_BINARY_DIR}/${entry}.ll
              -o ${CMAKE_CURRENT_BINARY_DIR}/${entry}.o
      DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${entry}.ll
      COMMENT "[rouge-compiler] clang: ${entry}.ll -> native object (AOT)")
    list(APPEND MMA_FORMS_OBJS ${CMAKE_CURRENT_BINARY_DIR}/${entry}.o)
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

  # Test 2k: narrow-float packed conversions (fp8_cvt kernel, AOT vs interpreter).
  add_executable(test_aot_fp8_cvt test_aot_fp8_cvt.cpp
                 ${CMAKE_CURRENT_BINARY_DIR}/fp8_cvt.o)
  target_link_libraries(test_aot_fp8_cvt PRIVATE rouge-runtime rouge-ptx)
  target_include_directories(test_aot_fp8_cvt PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../rouge-ptx)
  add_test(NAME aot_native_fp8_cvt COMMAND test_aot_fp8_cvt
           ${CMAKE_CURRENT_SOURCE_DIR}/kernels/fp8_cvt.ptx)

  # Test 2k-a: byte permute and byte dot products (prmt/dp4a/dp2a), AOT vs
  # interpreter. The semantics follow the PTX ISA; the same kernel is checked
  # on a real card by compiler_nvptx_gpu_exec.
  # Test 2j-bis: bfi/bfe/prmt pinned to the PTX ISA. Three-way comparison:
  # the native code, the interpreter, and the ISA pseudo-code written out in
  # the test. Two paths agreeing is not enough — they agreed on a wrong
  # answer once.
  add_executable(test_aot_bitfield test_aot_bitfield.cpp
                 ${CMAKE_CURRENT_BINARY_DIR}/bitfield.o)
  target_link_libraries(test_aot_bitfield PRIVATE rouge-runtime rouge-ptx)
  target_include_directories(test_aot_bitfield PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../rouge-ptx)
  add_test(NAME aot_native_bitfield COMMAND test_aot_bitfield
           ${CMAKE_CURRENT_SOURCE_DIR}/kernels/bitfield.ptx)

  add_executable(test_aot_prmt_dp4a test_aot_prmt_dp4a.cpp
                 ${CMAKE_CURRENT_BINARY_DIR}/prmt_dp4a.o)
  target_link_libraries(test_aot_prmt_dp4a PRIVATE rouge-runtime rouge-ptx)
  target_include_directories(test_aot_prmt_dp4a PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../rouge-ptx)
  add_test(NAME aot_native_prmt_dp4a COMMAND test_aot_prmt_dp4a
           ${CMAKE_CURRENT_SOURCE_DIR}/kernels/prmt_dp4a.ptx)

  # Test 2k-a: signed vs unsigned comparisons (cmp_signed kernel): the
  # lowering must not confuse setp.lt.s32 with setp.lt.u32 — it did, and the
  # differential fuzzer found it.
  add_executable(test_aot_cmp_signed ${CMAKE_CURRENT_SOURCE_DIR}/test_aot_cmp_signed.cpp
                 ${CMAKE_CURRENT_BINARY_DIR}/cmp_signed.o)
  target_link_libraries(test_aot_cmp_signed PRIVATE rouge-runtime rouge-ptx)
  target_include_directories(test_aot_cmp_signed PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../rouge-ptx)
  add_test(NAME aot_native_cmp_signed COMMAND test_aot_cmp_signed
           ${CMAKE_CURRENT_SOURCE_DIR}/kernels/cmp_signed.ptx)

  # Test 2k-bis: INT8 GEMM (nvcc-compiled kernel, the prmt + dp4a path),
  # AOT vs interpreter. Integer arithmetic, so the check is bit-exact.
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/int8_gemm.ll
    COMMAND ptx2ir ${CMAKE_CURRENT_SOURCE_DIR}/kernels/cuda/int8_gemm.ptx
            ${CMAKE_CURRENT_BINARY_DIR}/int8_gemm.ll
    DEPENDS ptx2ir ${CMAKE_CURRENT_SOURCE_DIR}/kernels/cuda/int8_gemm.ptx
    COMMENT "[rouge-compiler] ptx2ir: cuda/int8_gemm.ptx -> int8_gemm.ll")
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/int8_gemm.o
    COMMAND ${CLANG_EXECUTABLE} -O2 -c ${CMAKE_CURRENT_BINARY_DIR}/int8_gemm.ll
            -o ${CMAKE_CURRENT_BINARY_DIR}/int8_gemm.o
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/int8_gemm.ll
    COMMENT "[rouge-compiler] clang: int8_gemm.ll -> native object (AOT)")
  add_executable(test_aot_int8_gemm test_aot_int8_gemm.cpp
                 ${CMAKE_CURRENT_BINARY_DIR}/int8_gemm.o)
  target_link_libraries(test_aot_int8_gemm PRIVATE rouge-runtime rouge-ptx)
  target_include_directories(test_aot_int8_gemm PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../rouge-ptx)
  add_test(NAME aot_native_int8_gemm COMMAND test_aot_int8_gemm
           ${CMAKE_CURRENT_SOURCE_DIR}/kernels/cuda/int8_gemm.ptx)

  # Test 2m: mma.sync through the per-block exchange region, AOT vs
  # interpreter vs the PTX ISA lane formulas.
  add_executable(test_aot_mma_tile test_aot_mma_tile.cpp
                 ${CMAKE_CURRENT_BINARY_DIR}/mma_tile.o)
  target_link_libraries(test_aot_mma_tile PRIVATE rouge-runtime rouge-ptx)
  target_include_directories(test_aot_mma_tile PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../rouge-ptx)
  add_test(NAME aot_native_mma_tile COMMAND test_aot_mma_tile
           ${CMAKE_CURRENT_SOURCE_DIR}/kernels/mma_tile.ptx)

  # Test 2n: ldmatrix/stmatrix x4 roundtrip through the address exchange.
  add_executable(test_aot_ldmatrix_tile test_aot_ldmatrix_tile.cpp
                 ${CMAKE_CURRENT_BINARY_DIR}/ldmatrix_tile.o
                 ${CMAKE_CURRENT_BINARY_DIR}/ldmatrix_trans.o
                 ${CMAKE_CURRENT_BINARY_DIR}/ldmatrix_x4.o
                 ${CMAKE_CURRENT_BINARY_DIR}/ldmatrix_sm75.o
                 ${CMAKE_CURRENT_BINARY_DIR}/stmatrix_sm90.o)
  target_link_libraries(test_aot_ldmatrix_tile PRIVATE rouge-runtime rouge-ptx)
  target_include_directories(test_aot_ldmatrix_tile PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../rouge-ptx)
  add_test(NAME aot_native_ldmatrix_tile COMMAND test_aot_ldmatrix_tile
           ${CMAKE_CURRENT_SOURCE_DIR}/kernels/ldmatrix_tile.ptx)

  # Test 2o: every supported mma.sync form — ones against hand bits,
  # lane-varying fragments AOT against the interpreter.
  add_executable(test_aot_mma_forms test_aot_mma_forms.cpp ${MMA_FORMS_OBJS})
  target_link_libraries(test_aot_mma_forms PRIVATE rouge-runtime rouge-ptx)
  target_include_directories(test_aot_mma_forms PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../rouge-ptx)
  add_test(NAME aot_native_mma_forms COMMAND test_aot_mma_forms
           ${CMAKE_CURRENT_SOURCE_DIR}/kernels/mma_forms.ptx)

  # Test 2p: the full tensor pipeline (fill -> ldmatrix.x4 + x2.trans ->
  # mma -> global), AOT against the interpreter and the hand tile reference.
  add_executable(test_aot_mma_pipe test_aot_mma_pipe.cpp
                 ${CMAKE_CURRENT_BINARY_DIR}/mma_pipe.o)
  target_link_libraries(test_aot_mma_pipe PRIVATE rouge-runtime rouge-ptx)
  target_include_directories(test_aot_mma_pipe PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../rouge-ptx)
  add_test(NAME aot_native_mma_pipe COMMAND test_aot_mma_pipe
           ${CMAKE_CURRENT_SOURCE_DIR}/kernels/mma_pipe.ptx)

  # Benchmark (not a check): AOT against the interpreter on the collective
  # kernels, steady team of 32 threads. Always exits 0; correctness belongs
  # to test_aot_*. Links the already built kernel objects.
  add_executable(bench_collective bench_collective.cpp
                 ${CMAKE_CURRENT_BINARY_DIR}/mma_pipe.o
                 ${CMAKE_CURRENT_BINARY_DIR}/ldmatrix_x4.o)
  target_link_libraries(bench_collective PRIVATE rouge-runtime rouge-ptx)
  target_include_directories(bench_collective PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../rouge-ptx)
  add_test(NAME bench_collective COMMAND bench_collective
           ${CMAKE_CURRENT_SOURCE_DIR}/kernels)

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

  # Test 2h: predicated shfl.sync.down (scalar identity values + SET-predicates).
  add_executable(test_aot_shfl_pred test_aot_shfl_pred.cpp
                 ${CMAKE_CURRENT_BINARY_DIR}/shfl_pred.o)
  target_link_libraries(test_aot_shfl_pred PRIVATE rouge-runtime)
  add_test(NAME aot_native_shfl_pred COMMAND test_aot_shfl_pred)

  # Test 2i: vote.sync.any/all/uni + activemask (scalar identity + all-ones).
  add_executable(test_aot_vote_pred test_aot_vote_pred.cpp
                 ${CMAKE_CURRENT_BINARY_DIR}/vote_pred.o)
  target_link_libraries(test_aot_vote_pred PRIVATE rouge-runtime)
  add_test(NAME aot_native_vote_pred COMMAND test_aot_vote_pred)

  # Test 2j: predicated up/bfly/idx + legacy predicated down.
  add_executable(test_aot_shfl_pred_all test_aot_shfl_pred_all.cpp
                 ${CMAKE_CURRENT_BINARY_DIR}/shfl_pred_all.o)
  target_link_libraries(test_aot_shfl_pred_all PRIVATE rouge-runtime)
  add_test(NAME aot_native_shfl_pred_all COMMAND test_aot_shfl_pred_all)

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

  # ---- Test: Tier-0 foreign-PTX subset sweep (no GPU, no nvcc needed) ----
  # Small probes for vendor-PTX features (barrier.sync spelling, .volatile,
  # generic ld/st, .local depots, dynamic .shared, %clock64 fail-loud) go
  # through ptx2ir + clang for host and NVPTX. Any rejection is a bug,
  # except probe_clock on host, which must refuse loudly by design.
  add_test(NAME compiler_foreign_ptx
    COMMAND ${CMAKE_COMMAND}
      -DCLANG=${CLANG_EXECUTABLE}
      -DPTX2IR=$<TARGET_FILE:ptx2ir>
      -DKERNEL_DIR=${CMAKE_CURRENT_SOURCE_DIR}/kernels
      -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/foreign-ptx
      -DSM=sm_75
      -P ${CMAKE_CURRENT_SOURCE_DIR}/foreign_ptx.cmake)

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

  # ---- Test 3b: the same IR is not just compiled but RUN on RISC-V ----
  # qemu-user executes the freestanding riscv64 binary; the kernel's result
  # is checked there. Skips honestly when qemu-riscv64 is absent.
  find_program(ROUGE_QEMU_RISCV64 NAMES qemu-riscv64 qemu-riscv64-static)
  if(ROUGE_QEMU_RISCV64)
    add_test(NAME compiler_rv_exec
      COMMAND ${CMAKE_COMMAND}
        -DCLANG=${CLANG_EXECUTABLE}
        -DPTX2IR=$<TARGET_FILE:ptx2ir>
        -DKERNEL=${CMAKE_CURRENT_SOURCE_DIR}/kernels/vadd.ptx
        -DHARNESS=${CMAKE_CURRENT_SOURCE_DIR}/rv_exec_harness.c
        -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/rv-exec
        -DQEMU=${ROUGE_QEMU_RISCV64}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/rv_exec.cmake)
  else()
    message(STATUS "compiler_rv_exec: SKIP (qemu-riscv64 not found)")
  endif()

  # ---- Test 3c: a block kernel with bar.sync, with REAL threads, on RISC-V --
  # Needs a riscv64 sysroot (ROUGE_RISCV64_SYSROOT); without it the test says
  # SKIP honestly. Recipe for the sysroot is in the script header.
  find_program(ROUGE_CLANGXX NAMES clang++ g++)
  if(ROUGE_QEMU_RISCV64 AND ROUGE_CLANGXX AND ROUGE_RISCV64_SYSROOT)
    add_test(NAME compiler_rv_exec_mt
      COMMAND ${CMAKE_COMMAND}
        -DCLANG=${CLANG_EXECUTABLE}
        -DCLANGXX=${ROUGE_CLANGXX}
        -DPTX2IR=$<TARGET_FILE:ptx2ir>
        -DKERNEL=${CMAKE_CURRENT_SOURCE_DIR}/kernels/block_reduce.ptx
        -DHARNESS=${CMAKE_CURRENT_SOURCE_DIR}/rv_exec_mt.c
        -DRUNTIME=${CMAKE_CURRENT_SOURCE_DIR}/../runtime/rouge_runtime.cpp
        -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/rv-exec-mt
        -DQEMU=${ROUGE_QEMU_RISCV64}
        -DSYSROOT=${ROUGE_RISCV64_SYSROOT}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/rv_exec_mt.cmake)
  else()
    message(STATUS "compiler_rv_exec_mt: SKIP (needs qemu-riscv64 + "
                   "-DROUGE_RISCV64_SYSROOT=<prefix>)")
  endif()

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
  # Warp-collective matrix ops need newer architectures than the base matrix:
  # ldmatrix is sm_75, mma.sync m16n8k8/m16n8k16 is sm_80, stmatrix is sm_90.
  # These cells pin the arch explicitly; without a ptxas that knows it the
  # backend probe still passes (it only probes the triple) and clang fails the
  # cell loudly instead of silently skipping it.
  set(ROUGE_NVPTX_SM80_FLAGS "-march=sm_80 -O2" CACHE STRING
      "Extra clang flags for the sm_80 NVIDIA matrix cells (mma.sync)")
  set(ROUGE_NVPTX_SM90_FLAGS "-march=sm_90 -O2" CACHE STRING
      "Extra clang flags for the sm_90 NVIDIA matrix cells (stmatrix)")

  # These values are handed to the matrix script as single -D arguments, where a
  # ';' or a space would be split by the CTest command line. The script therefore
  # expects ',' between list items and '+' between the clang flags of one spec.
  string(REPLACE " " "+" AMDGPU_FLAGS_SPEC "${ROUGE_AMDGPU_FLAGS}")
  string(REPLACE " " "+" NVPTX_FLAGS_SPEC "${ROUGE_NVPTX_FLAGS}")
  string(REPLACE " " "+" NVPTX_SM80_FLAGS_SPEC "${ROUGE_NVPTX_SM80_FLAGS}")
  string(REPLACE " " "+" NVPTX_SM90_FLAGS_SPEC "${ROUGE_NVPTX_SM90_FLAGS}")
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

  # Warp-collective matrix cells per architecture: ldmatrix assembles from
  # sm_75 (and runs there — see compiler_nvptx_gpu_exec), mma.sync m16n8k8/k16
  # from sm_80, stmatrix from sm_90. mma/stmatrix cannot execute on tensor-less
  # silicon, so these cells verify translation + backend + ptxas acceptance
  # (clang -c shells out to ptxas); execution stays loudly refused/uncovered.
  add_test(NAME compiler_nvptx_backend_collective_sm75
    COMMAND ${CMAKE_COMMAND}
      -DCLANG=${CLANG_EXECUTABLE}
      -DPTX2IR=$<TARGET_FILE:ptx2ir>
      -DKERNEL_DIR=${CMAKE_CURRENT_SOURCE_DIR}/kernels
      -DIR_FILE=${CMAKE_CURRENT_BINARY_DIR}/vadd.ll
      -DOUT_FILE=${CMAKE_CURRENT_BINARY_DIR}/vadd-nvptx.o
      -DKERNELS=ldmatrix_tile@ldmatrix_sm75
      -DPLATFORMS=nvptx|${ROUGE_NVPTX_TRIPLE}|${NVPTX_FLAGS_SPEC}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/gpu_targets.cmake)
  add_test(NAME compiler_nvptx_backend_collective_sm80
    COMMAND ${CMAKE_COMMAND}
      -DCLANG=${CLANG_EXECUTABLE}
      -DPTX2IR=$<TARGET_FILE:ptx2ir>
      -DKERNEL_DIR=${CMAKE_CURRENT_SOURCE_DIR}/kernels
      -DIR_FILE=${CMAKE_CURRENT_BINARY_DIR}/vadd.ll
      -DOUT_FILE=${CMAKE_CURRENT_BINARY_DIR}/vadd-nvptx.o
      -DKERNELS=mma_tile,mma_pipe
      -DPLATFORMS=nvptx|${ROUGE_NVPTX_TRIPLE}|${NVPTX_SM80_FLAGS_SPEC}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/gpu_targets.cmake)
  add_test(NAME compiler_nvptx_backend_collective_sm90
    COMMAND ${CMAKE_COMMAND}
      -DCLANG=${CLANG_EXECUTABLE}
      -DPTX2IR=$<TARGET_FILE:ptx2ir>
      -DKERNEL_DIR=${CMAKE_CURRENT_SOURCE_DIR}/kernels
      -DIR_FILE=${CMAKE_CURRENT_BINARY_DIR}/vadd.ll
      -DOUT_FILE=${CMAKE_CURRENT_BINARY_DIR}/vadd-nvptx.o
      -DKERNELS=ldmatrix_tile@stmatrix_sm90
      -DPLATFORMS=nvptx|${ROUGE_NVPTX_TRIPLE}|${NVPTX_SM90_FLAGS_SPEC}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/gpu_targets.cmake)
endif()

# ---- Tests 7 + 8: on-device execution on a real NVIDIA GPU ----
# The kernels our translator produced are loaded through the CUDA Driver API
# (driver-side JIT) and checked bit-exactly against a host reference. Needs
# the CUDA driver headers + libcuda at build time and a real sm_75+ NVIDIA GPU
# at test time; everywhere else the tests report SKIP and never fail the run.
find_path(CUDA_DRIVER_INCLUDE_DIR cuda.h
  PATHS /opt/cuda/targets/x86_64-linux/include /usr/local/cuda/include
  NO_DEFAULT_PATH)
find_library(CUDA_DRIVER_LIBRARY NAMES cuda
  PATHS /usr/lib /usr/lib64 /usr/lib/x86_64-linux-gnu)
if(CLANG_EXECUTABLE AND CUDA_DRIVER_INCLUDE_DIR AND CUDA_DRIVER_LIBRARY)
  # The project is CXX-only; the runners are plain C.
  enable_language(C)
  add_executable(gpu_exec_rungpu gpu_exec_rungpu.c)
  target_include_directories(gpu_exec_rungpu PRIVATE ${CUDA_DRIVER_INCLUDE_DIR})
  target_link_libraries(gpu_exec_rungpu PRIVATE ${CUDA_DRIVER_LIBRARY} m)
  add_executable(gpu_exec_runatomic gpu_exec_runatomic.c)
  target_include_directories(gpu_exec_runatomic PRIVATE ${CUDA_DRIVER_INCLUDE_DIR})
  target_link_libraries(gpu_exec_runatomic PRIVATE ${CUDA_DRIVER_LIBRARY} m)
  add_executable(gpu_exec_runreal gpu_exec_runreal.c)
  target_include_directories(gpu_exec_runreal PRIVATE ${CUDA_DRIVER_INCLUDE_DIR})
  target_link_libraries(gpu_exec_runreal PRIVATE ${CUDA_DRIVER_LIBRARY})
  add_executable(gpu_exec_vendor gpu_exec_vendor.c)
  target_include_directories(gpu_exec_vendor PRIVATE ${CUDA_DRIVER_INCLUDE_DIR})
  target_link_libraries(gpu_exec_vendor PRIVATE ${CUDA_DRIVER_LIBRARY} m)

  # Tier-1 vendor kernels: genuine nvcc output in kernels/vendor/*.ptx, run
  # through OUR NVPTX lowering and checked on the card. Multi-entry modules,
  # so each cell names its entry explicitly.
  add_test(NAME compiler_nvptx_vendor_gpu_exec
    COMMAND ${CMAKE_COMMAND}
      -DCLANG=${CLANG_EXECUTABLE}
      -DPTX2IR=$<TARGET_FILE:ptx2ir>
      -DRUNNER=$<TARGET_FILE:gpu_exec_vendor>
      -DKERNEL_DIR=${CMAKE_CURRENT_SOURCE_DIR}/kernels
      -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/vendor-gpu-exec
      "-DCELLS=vecAdd.VecAdd_kernel,matrixMul._Z13MatrixMulCUDAILi16EEvPfS0_S0_ii,transpose.transposeCoalesced,reduction.blockSum,threadFence.fenceProducer"
      -DSM=sm_75
      -P ${CMAKE_CURRENT_SOURCE_DIR}/vendor_gpu.cmake)
  set_tests_properties(compiler_nvptx_vendor_gpu_exec PROPERTIES
    SKIP_REGULAR_EXPRESSION "NO-DEVICE|SKIP-OLD-ARCH")

  # kernel.selector cells: dataset A for fp16/gemm, dataset B, a two-block grid
  # proving the nctaid path, warp shuffles, and cas/exch/min/max atomics.
  add_test(NAME compiler_nvptx_gpu_exec
    COMMAND ${CMAKE_COMMAND}
      -DCLANG=${CLANG_EXECUTABLE}
      -DPTX2IR=$<TARGET_FILE:ptx2ir>
      -DRUNNER=$<TARGET_FILE:gpu_exec_rungpu>
      -DKERNEL_DIR=${CMAKE_CURRENT_SOURCE_DIR}/kernels
      -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/gpu-exec
      "-DCELLS=vadd.vadd,prmt_dp4a.prmt_dp4a,cmp_signed.cmp_signed,shift_clamp.shift_clamp,block_reduce.block_reduce,fp16_reduce.fp16_reduce,fp16_reduce.fp16_reduceB,gemm_tile.gemm_tile,gemm_tile.gemm_tileB,gemm_tile.gemm_tile2,shfl_reduce.shfl_reduce,atom_cas.atom_cas,shfl_pred.shfl_pred,vote_pred.vote_pred,shfl_pred_all.shfl_pred_all,ldmatrix_tile.ldmatrix_sm75@ldmatrix_sm75"
      -DSM=sm_75
      -P ${CMAKE_CURRENT_SOURCE_DIR}/gpu_exec.cmake)
  set_tests_properties(compiler_nvptx_gpu_exec PROPERTIES
    SKIP_REGULAR_EXPRESSION "NO-DEVICE|SKIP-OLD-ARCH")

  # Our own CUDA C kernel (kernels/cuda/int8_gemm.cu) compiled by the user's
  # nvcc into PTX, then translated by OUR ptx2ir and run on the card by OUR
  # runtime. The arithmetic is integer, so the check is bit-exact: the GPU
  # answer must equal the host's int32 accumulation of the same product.
  # Its .ptx is committed (like the vendor kernels); regenerate with
  #   nvcc -ptx -arch=sm_75 -O2 kernels/cuda/int8_gemm.cu -o kernels/cuda/int8_gemm.ptx
  add_test(NAME compiler_nvptx_cuda_gpu_exec
    COMMAND ${CMAKE_COMMAND}
      -DCLANG=${CLANG_EXECUTABLE}
      -DPTX2IR=$<TARGET_FILE:ptx2ir>
      -DRUNNER=$<TARGET_FILE:gpu_exec_rungpu>
      -DKERNEL_DIR=${CMAKE_CURRENT_SOURCE_DIR}/kernels
      -DSUBDIR=cuda
      -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/cuda-gpu-exec
      "-DCELLS=int8_gemm.int8_gemm"
      -DSM=sm_75
      -P ${CMAKE_CURRENT_SOURCE_DIR}/gpu_exec.cmake)
  set_tests_properties(compiler_nvptx_cuda_gpu_exec PROPERTIES
    SKIP_REGULAR_EXPRESSION "NO-DEVICE|SKIP-OLD-ARCH")

  add_test(NAME compiler_nvptx_gpu_exec_atomic
    COMMAND ${CMAKE_COMMAND}
      -DCLANG=${CLANG_EXECUTABLE}
      -DPTX2IR=$<TARGET_FILE:ptx2ir>
      -DRUNNER=$<TARGET_FILE:gpu_exec_runatomic>
      -DKERNEL_DIR=${CMAKE_CURRENT_SOURCE_DIR}/kernels
      -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/gpu-exec
      -DCELLS=atomic_reduce.atomic_reduce
      -DSM=sm_75
      -P ${CMAKE_CURRENT_SOURCE_DIR}/gpu_exec.cmake)
  set_tests_properties(compiler_nvptx_gpu_exec_atomic PROPERTIES
    SKIP_REGULAR_EXPRESSION "NO-DEVICE|SKIP-OLD-ARCH")

  # Vendor end-to-end: nvcc -> ptx2ir -> GPU, bit-exact. Needs nvcc at test
  # time; without it (or without a usable GPU) the script reports SKIP.
  find_program(NVCC_EXECUTABLE nvcc)
  add_test(NAME compiler_nvcc_gpu_exec
    COMMAND ${CMAKE_COMMAND}
      -DNVCC=${NVCC_EXECUTABLE}
      -DCLANG=${CLANG_EXECUTABLE}
      -DPTX2IR=$<TARGET_FILE:ptx2ir>
      -DRUNNER=$<TARGET_FILE:gpu_exec_runreal>
      -DCU_SRC=${CMAKE_CURRENT_SOURCE_DIR}/real_nvcc.cu
      -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/nvcc-gpu-exec
      -DSM=sm_75
      -P ${CMAKE_CURRENT_SOURCE_DIR}/nvcc_gpu_exec.cmake)
  set_tests_properties(compiler_nvcc_gpu_exec PROPERTIES
    SKIP_REGULAR_EXPRESSION "NO-DEVICE|SKIP-OLD-ARCH")
else()
  message(STATUS "[rouge-compiler] GPU exec tests disabled: need clang plus "
                 "the CUDA driver headers and libcuda")
endif()

# ---- Differential fuzzer (our compiler vs the real hardware) --------------
add_subdirectory(fuzz)
