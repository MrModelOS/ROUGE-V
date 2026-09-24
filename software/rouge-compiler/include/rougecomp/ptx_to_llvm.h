#pragma once

#include <string>

#include "ptx.h"

namespace rougecomp {

// Translate a parsed PTX kernel (prog.functions[fnIndex]) into textual LLVM IR
// for the ROUGE-V AOT path: the kernel body is compiled to native code instead
// of being emulated instruction-by-instruction.
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
                           std::string* error);

}  // namespace rougecomp