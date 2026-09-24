// Structural test for the atomics + FP16/BF16 lowering and the exported
// .shared metadata in ptx2ir.
//
// Checks that:
//   * atom.add.* / red.add.* lower to LLVM atomicrmw (monotonic = relaxed),
//     with the returning form feeding the destination register;
//   * f16/bf16 arithmetic widens to f32, computes, and narrows with fptrunc;
//     fma becomes the llvm.fma.f32 intrinsic call;
//   * 16-bit PTX registers become i16 allocas;
//   * the .shared layout is exported as kernel metadata (@__rouge_*_query),
//     so drivers never hard-code the scratchpad size.

#include <cstdio>
#include <string>

#include "ptx.h"
#include "rougecomp/ptx_to_llvm.h"

namespace {

int failures = 0;

void expect(const std::string& ir, const char* needle, const char* what) {
  if (ir.find(needle) == std::string::npos) {
    std::printf("FAIL: %s — missing \"%s\"\n", what, needle);
    ++failures;
  } else {
    std::printf("ok:   %s\n", what);
  }
}

const char* kPtx = R"(
.version 8.0
.target sm_80
.address_size 64

.visible .entry probe(
    .param .u64 probe_param_0,
    .param .u64 probe_param_1
)
{
    .reg .b32 %r<6>;
    .reg .b16 %h<6>;
    .reg .b64 %rd<4>;
    .reg .pred %p<1>;
    .shared .align 2 .b16 tile[16];
    .shared .align 4 .f32 acc[8];

    ld.param.u64 %rd1, [probe_param_0];
    ld.param.u64 %rd2, [probe_param_1];

    // relaxed atomics: returning integer form and result-less f32 form
    mov.u32 %r1, 1;
    atom.global.add.u32 %r2, [%rd1], %r1;
    ld.global.f32 %r3, [%rd1];
    red.global.add.f32 [%rd1], %r3;

    // f16 pipeline
    ld.global.f16 %h1, [%rd1];
    add.f16 %h2, %h1, %h1;
    fma.rn.f16 %h3, %h2, %h2, %h1;
    cvt.f32.f16 %r4, %h3;
    cvt.rn.f16.f32 %h4, %r4;
    cvt.rn.bf16.f32 %h1, %r4;
    cvt.f32.bf16 %r5, %h1;

    st.global.f16 [%rd2], %h4;
    ret;
}
)";

}  // namespace

int main() {
  std::string err;
  auto prog = rouge::parse_ptx(kPtx, &err);
  if (!prog || prog->functions.empty()) {
    std::printf("FAIL: cannot parse probe.ptx: %s\n", err.c_str());
    return 1;
  }
  const rouge::PtxFunction& fn = prog->functions[0];
  if (fn.sharedVars.size() != 2) {
    std::printf("FAIL: expected 2 .shared vars, got %zu\n", fn.sharedVars.size());
    return 1;
  }
  std::printf("ok:   .shared layout — tile@%d, acc@%d, scratchpad=%d bytes\n",
              fn.sharedVars[0].offset, fn.sharedVars[1].offset, fn.sharedSize);

  err.clear();
  const std::string ir = rougecomp::ptx_to_llvm_ir(*prog, 0, &err);
  if (ir.empty()) {
    std::printf("FAIL: translator error: %s\n", err.c_str());
    return 1;
  }

  // --- atomics ---
  expect(ir, "= atomicrmw add ptr", "atom.add.u32 -> atomicrmw add");
  expect(ir, "monotonic", "atomics are relaxed (monotonic)");
  expect(ir, "= atomicrmw fadd ptr", "red.add.f32 -> atomicrmw fadd");
  expect(ir, "bitcast float", "atomic f32 result reinterpreted as i32");

  // --- f16 / bf16 ---
  expect(ir, "= alloca i16", "16-bit PTX registers -> i16 allocas");
  expect(ir, "= bitcast i16 ", "i16 <-> half/bfloat bitcasts");
  expect(ir, "= fpext half ", "f16 widened to f32");
  expect(ir, "= fptrunc float ", "f32 narrowed back (RNE)");
  expect(ir, "bfloat", "bfloat type used");
  expect(ir, "@llvm.fma.f32(", "fma.rn.f16 -> llvm.fma intrinsic");

  // --- exported .shared metadata ---
  expect(ir, "@__rouge_probe_shared_vars", "shared layout table exported");
  expect(ir, "define i32 @__rouge_probe_query(ptr %info)",
         "kernel info query emitted");
  expect(ir, "c\"tile\\00\"", "shared symbol names exported");
  expect(ir, "c\"acc\\00\"", "second shared symbol exported");

  if (failures) {
    std::printf("%d structural checks FAILED\n", failures);
    return 1;
  }
  std::printf("all atomics/fp16/metadata structural checks passed\n");
  return 0;
}
