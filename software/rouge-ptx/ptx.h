#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace rouge {

struct PtxInstruction {
  std::string op;                  // mnemonic, e.g. "ld.global.f32"
  std::string pred;                // predicate reg, e.g. "p1" or "!p1" ("" if none)
  std::vector<std::string> args;   // normalized operands: "r1", "[rd2]", "tid.x", "LBB0_2", "4"
};

struct PtxParam {
  std::string name;
  int width = 0;   // 4 or 8 bytes
  int index = 0;   // order in the kernel signature (kernelParams order)
  int offset = 0;  // byte offset in the bound-params blob
};

// A .shared variable: per-block scratchpad memory.
// Layout: offset within the block's scratchpad, filled by the layout pass in
// parse_ptx (alignment-padded, see docs/06-compiler-architecture.md).
struct PtxSharedVar {
  std::string name;
  int align = 4;      // declared alignment (e.g. .align 4)
  int elemBytes = 4;  // element size: .b8=1, .b16=2, .b32/.f32=4, .b64/.f64=8
  int count = 1;      // [N] elements
  int offset = 0;     // byte offset in the block scratchpad
};

// A module-scope global: ".global .align 4 .b32 arr[4096];"
// Real nvcc output declares every __device__ variable this way, and
// "cvta.to.global.u64 %rd, arr" needs the symbol's real address, so the
// declaration has to be carried into the backend.
struct PtxGlobalVar {
  std::string name;
  int align = 4;
  int elemBytes = 4;  // 1, 2, 4 or 8
  int count = 1;      // [N] elements
  bool isConst = false;
};

struct PtxFunction {
  std::string name;
  std::vector<PtxParam> params;
  std::vector<PtxInstruction> body;
  std::unordered_map<std::string, int> labels;  // label -> body index
  // Register declarations from ".reg .b32 %r<9>;" lines:
  // key = register name without '%' ("r0", "rd1", "p2"),
  // value = width in bits: 1 for .pred, 32 for 32-bit, 64 for 64-bit.
  std::unordered_map<std::string, int> regBits;
  // .shared variables with laid-out byte offsets; sharedSize = scratchpad bytes.
  std::vector<PtxSharedVar> sharedVars;
  std::unordered_map<std::string, int> sharedIndex;  // name -> index in sharedVars
  int sharedSize = 0;
  int paramsBlobSize = 0;
};

struct PtxProgram {
  std::vector<PtxFunction> functions;
  std::unordered_map<std::string, int> functionIndex;  // name -> index
  std::vector<PtxGlobalVar> globals;                    // .global/.const decls
  std::unordered_map<std::string, int> globalIndex;    // name -> index
};

// Parse PTX assembly text into a program.
std::unique_ptr<PtxProgram> parse_ptx(const std::string& text, std::string* error);

// Execute a kernel over the full grid for the CPU-emulated device.
bool execute_kernel(const PtxProgram& prog, int fnIndex,
                    const std::vector<uint8_t>& paramsBlob,
                    unsigned gx, unsigned gy, unsigned gz,
                    unsigned bx, unsigned by, unsigned bz,
                    std::string* error);

}  // namespace rouge