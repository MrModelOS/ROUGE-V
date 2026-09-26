#pragma once

#include <string>

#include "ptx.h"

namespace rougecomp {

// Translate a parsed PTX kernel (prog.functions[fnIndex]) into textual LLVM IR
// for the ROUGE-V AOT path: the kernel body is compiled to native code instead
// of being emulated instruction-by-instruction.
//
// `target` is an LLVM target triple that the emitted module declares, e.g.
//   "x86_64-pc-linux-gnu"          host execution and the reference tests
//   "riscv64-unknown-elf"          RISC-V, with -march=rv64gcv for the vectors
//   "amdgcn-amd-amdhsa"            AMD GPU (RDNA), with -mcpu=gfx1100
//   "nvptx64-nvidia-cuda"          NVIDIA GPU, with -march=sm_XX
// It selects the address spaces the backend needs (AMDGPU numbers shared
// memory 5, NVPTX 3) so the GPU backends emit real global_load/global_store
// rather than scalar accesses through integers.
//
// Emitted signature:
//   define void @<kernel>(i64/i32 %arg0, ..., ptr %launch)
// where %launch points to a launch descriptor with this layout:
//   offset  0..44: 12 x i32  (tid/ctaid/ntid/nctaid x y z)
//   offset 48   : i64  scratchpad base (block .shared memory)
//   offset 56   : i64  barrier handle (passed to __rouge_syncthreads)
// CUDA special registers (%ctaid.x and friends) are loaded from this structure
// by the host runtime before each kernel launch.
//
// .shared variables are laid out in the scratchpad (byte offsets computed by
// the parser); ld/st.shared and cvta.to.shared resolve symbols to
// scratchpadBase + offset. bar.sync lowers to a call to the runtime function
// __rouge_syncthreads (pthread_barrier on the host, fence on RISC-V).
//
// Returns the IR text, or "" with *error set when a PTX instruction is outside
// the currently supported subset.
std::string ptx_to_llvm_ir(const rouge::PtxProgram& prog, int fnIndex,
                           std::string* error,
                           const std::string& target = "x86_64-pc-linux-gnu");

}  // namespace rougecomp