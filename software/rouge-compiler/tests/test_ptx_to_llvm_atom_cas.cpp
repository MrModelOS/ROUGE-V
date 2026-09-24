// Structural test for atom.cas / exch / min / max lowering.

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
.visible .entry probe_cas(
    .param .u64 probe_param_0,
    .param .u64 probe_param_1
)
{
    .reg .b32 %r<10>;
    .reg .b64 %rd<4>;
    ld.param.u64 %rd1, [probe_param_0];
    ld.param.u64 %rd2, [probe_param_1];
    mov.u32 %r1, 42;
    mov.u32 %r2, 100;
    atom.global.cas.b32 %r3, [%rd1], %r1, %r2;
    mov.u32 %r4, 55;
    atom.global.exch.b32 %r5, [%rd1], %r4;
    mov.u32 %r6, 10;
    atom.global.min.s32 %r7, [%rd1], %r6;
    mov.u32 %r8, 90;
    atom.global.max.u32 %r9, [%rd1], %r8;
    // f32 cas via b32 bitcast
    mov.u32 %r1, 0;
    atom.global.cas.b32 %r3, [%rd2], %r1, %r2;
    ret;
}
)";
} // namespace

int main() {
  std::string err;
  auto prog = rouge::parse_ptx(kPtx, &err);
  if (!prog || prog->functions.empty()) {
    std::printf("FAIL: cannot parse probe_cas.ptx: %s\n", err.c_str());
    return 1;
  }
  err.clear();
  const std::string ir = rougecomp::ptx_to_llvm_ir(*prog, 0, &err);
  if (ir.empty()) {
    std::printf("FAIL: translator error: %s\n", err.c_str());
    return 1;
  }
  expect(ir, "cmpxchg ptr", "atom.cas -> cmpxchg");
  expect(ir, "extractvalue { i32, i1 }", "cmpxchg extract old value");
  expect(ir, "atomicrmw xchg ptr", "atom.exch -> atomicrmw xchg");
  expect(ir, "atomicrmw min ptr", "atom.min.s32 -> atomicrmw min");
  expect(ir, "atomicrmw umax ptr", "atom.max.u32 -> atomicrmw umax");
  expect(ir, "monotonic", "atomics are relaxed (monotonic)");
  if (failures) {
    std::printf("%d structural checks FAILED\n", failures);
    return 1;
  }
  std::printf("all cas/exch/min/max structural checks passed\n");
  return 0;
}
