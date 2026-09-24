// Structural test for warp shuffles + vote + activemask fallback lowering.
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

.visible .entry warp_probe(
    .param .u64 warp_param_0,
    .param .u64 warp_param_1
)
{
    .reg .b32 %r<8>;
    .reg .pred %p<4>;
    .reg .b64 %rd<3>;

    ld.param.u64 %rd1, [warp_param_0];
    ld.param.u64 %rd2, [warp_param_1];

    mov.u32 %r1, %tid.x;

    // shfl variants: bfly, idx, up, down (with and without .sync)
    shfl.bfly.b32 %r2, %r1, 16, 0x1f;
    shfl.sync.bfly.b32 %r2, %r1, 8, 0x1f, 0xffffffff;
    shfl.idx.b32 %r3, %r1, 5, 0x1f;
    shfl.sync.idx.b32 %r3, %r1, %r2, 0x1f, 0xffffffff;
    shfl.up.b32 %r4, %r1, 1, 0x1f;
    shfl.sync.down.b32 %r4, %r1, 2, 0x1f, 0xffffffff;

    // vote
    setp.eq.u32 %p1, %r1, 0;
    vote.any.pred %p2, %p1;
    vote.sync.all.pred %p3, %p1, 0xffffffff;
    vote.uni.pred %p2, %p1;

    // activemask
    activemask.b32 %r5;
    activemask.b32 %r6, %r1;

    // use results to keep them alive
    st.global.u32 [%rd1], %r2;
    st.global.u32 [%rd2], %r5;
    ret;
}
)";
} // namespace

int main() {
  std::string err;
  auto prog = rouge::parse_ptx(kPtx, &err);
  if (!prog || prog->functions.empty()) {
    std::printf("FAIL: cannot parse warp_probe.ptx: %s\n", err.c_str());
    return 1;
  }
  err.clear();
  std::string ir = rougecomp::ptx_to_llvm_ir(*prog, 0, &err);
  if (ir.empty()) {
    std::printf("FAIL: translator error: %s\n", err.c_str());
    return 1;
  }
  expect(ir, "scalar fallback: shfl is identity", "shfl lowered to scalar identity");
  expect(ir, "scalar fallback: vote", "vote lowered to scalar identity");
  expect(ir, "scalar fallback: activemask", "activemask lowered to all-ones");
  expect(ir, "store i32 -1", "activemask stores -1 (0xffffffff)");
  expect(ir, "ret void", "kernel ret");
  // shfl should not produce unsupported error
  if (ir.find("unsupported") != std::string::npos) {
    std::printf("FAIL: IR contains unsupported marker\n");
    ++failures;
  }
  std::printf("warp_probe LLVM IR: %zu bytes\n", ir.size());
  if (failures) {
    std::printf("%d structural checks FAILED\n", failures);
    return 1;
  }
  std::printf("all warp structural checks passed\n");
  return 0;
}
