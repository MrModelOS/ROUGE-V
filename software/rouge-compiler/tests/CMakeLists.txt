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
foreach(kern vadd block_reduce atomic_reduce fp16_reduce gemm_tile atom_cas shfl_reduce shfl_pred vote_pred shfl_pred_all fp8_cvt mma_tile)
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
  foreach(kern vadd block_reduce atomic_reduce fp16_reduce gemm_tile atom_cas shfl_reduce shfl_pred vote_pred shfl_pred_all fp8_cvt mma_tile)
    add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${kern}.o
      COMMAND ${CLANG_EXECUTABLE} -O2 -c ${CMAKE_CURRENT_BINARY_DIR}/${kern}.ll
              -o ${CMAKE_CURRENT_BINARY_DIR}/${kern}.o
      DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${kern}.ll
      COMMENT "[rouge-compiler] clang: ${kern}.ll -> native object (AOT)")
  endforeach()

  # The ldmatrix kernel has three entries, so ptx2ir runs once per entry (it
  # translates one .entry per module).
  foreach(entry ldmatrix_tile ldmatrix_trans ldmatrix_x4)
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
                 ${CMAKE_CURRENT_BINARY_DIR}/ldmatrix_x4.o)
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
      "-DCELLS=vadd.vadd,block_reduce.block_reduce,fp16_reduce.fp16_reduce,fp16_reduce.fp16_reduceB,gemm_tile.gemm_tile,gemm_tile.gemm_tileB,gemm_tile.gemm_tile2,shfl_reduce.shfl_reduce,atom_cas.atom_cas,shfl_pred.shfl_pred,vote_pred.vote_pred,shfl_pred_all.shfl_pred_all"
      -DSM=sm_75
      -P ${CMAKE_CURRENT_SOURCE_DIR}/gpu_exec.cmake)
  set_tests_properties(compiler_nvptx_gpu_exec PROPERTIES
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
