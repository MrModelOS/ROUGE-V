// Structural test for the PTX -> LLVM IR translator: parse the vadd kernel
// (the same one used by rouge-cuda) and check that the emitted IR contains
// the expected lowering for each PTX idiom. Fast test, no LLVM toolchain needed.

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

.visible .entry vadd(
    .param .u64 vadd_param_0,
    .param .u64 vadd_param_1,
    .param .u64 vadd_param_2,
    .param .u32 vadd_param_3
)
{
    .reg .b32 %r<7>;
    .reg .b64 %rd<6>;
    .reg .pred %p<2>;

    ld.param.u64 %rd1, [vadd_param_0];
    ld.param.u64 %rd2, [vadd_param_1];
    ld.param.u64 %rd3, [vadd_param_2];
    ld.param.u32 %r1, [vadd_param_3];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mul.wide.u32 %rd4, %r2, %r3;
    mov.u32 %r2, %tid.x;
    cvt.u64.u32 %rd5, %r2;
    add.s64 %rd4, %rd4, %rd5;
    cvt.u32.u64 %r4, %rd4;
    setp.ge.u32 %p1, %r4, %r1;
    @%p1 bra LBB0_2;
    mul.wide.u32 %rd5, %r4, 4;
    add.s64 %rd1, %rd1, %rd5;
    add.s64 %rd2, %rd2, %rd5;
    add.s64 %rd3, %rd3, %rd5;
    ld.global.f32 %r5, [%rd1];
    ld.global.f32 %r6, [%rd2];
    add.f32 %r7, %r5, %r6;
    st.global.f32 [%rd3], %r7;
LBB0_2:
    ret;
}
)";

}  // namespace

int main() {
  std::string err;
  auto prog = rouge::parse_ptx(kPtx, &err);
  if (!prog || prog->functions.empty()) {
    std::printf("FAIL: cannot parse vadd.ptx: %s\n", err.c_str());
    return 1;
  }
  err.clear();
  std::string ir = rougecomp::ptx_to_llvm_ir(*prog, 0, &err);
  if (ir.empty()) {
    std::printf("FAIL: translator error: %s\n", err.c_str());
    return 1;
  }

  // Kernel signature with the launch descriptor.
  expect(ir, "define void @vadd(i64 %arg0, i64 %arg1, i64 %arg2, i32 %arg3, ptr %launch)",
         "kernel signature (params + launch descriptor)");
  // Registers of all widths as allocas in the entry block.
  expect(ir, "alloca i64", "64-bit register allocas");
  expect(ir, "alloca i1", "predicate register allocas");
  // Special-register reads go through the launch descriptor (ctaid.x @ offset 12).
  expect(ir, "getelementptr inbounds i8, ptr %launch, i64 12", "ctaid.x from launch");
  // mul.wide.u32 -> zext pair + i64 mul.
  expect(ir, "= zext i32 ", "mul.wide.u32 zext");
  expect(ir, "= mul i64 ", "mul.wide.u32 i64 multiply");
  // cvt.u64.u32 -> zext; cvt.u32.u64 -> trunc.
  expect(ir, "= zext i32 ", "cvt.u64.u32");
  expect(ir, "= trunc i64 ", "cvt.u32.u64");
  // setp.ge.u32 -> icmp uge, predicated branch -> br i1.
  expect(ir, "= icmp uge i32 ", "setp.ge.u32 -> icmp uge");
  expect(ir, "br i1 ", "@p bra -> conditional branch");
  // Global memory through inttoptr + typed loads/stores (f32 via bitcast).
  expect(ir, "inttoptr i64 ", "address conversion");
  expect(ir, "= bitcast i32 ", "f32 value bitcast");
  expect(ir, "= fadd float ", "add.f32 -> fadd");
  // Control flow: fallthrough + label + ret.
  expect(ir, "br label %b_LBB0_2", "branch to label block");
  expect(ir, "ret void", "kernel ret");
  expect(ir, "LBB0_2:", "label block emitted");

  std::printf("vadd LLVM IR: %zu bytes\n", ir.size());
  if (failures) {
    std::printf("%d structural checks FAILED\n", failures);
    return 1;
  }
  std::printf("all structural checks passed\n");
  return 0;
}