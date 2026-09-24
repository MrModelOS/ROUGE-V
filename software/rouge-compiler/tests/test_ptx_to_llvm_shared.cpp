// Structural test for shared memory + bar.sync lowering in ptx2ir.
// Uses two .shared variables (s0 at offset 0, s1 at offset 32 — laid out by
// the parser) and checks that the emitted IR reads the scratchpad base from
// the launch descriptor, computes the right offsets, and lowers bar.sync to a
// runtime barrier call.

#include <cstdio>
#include <string>
#include <vector>

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

// Look for a line that contains both fragments (e.g. an `add i64` with a
// specific offset constant).
bool expect_line(const std::string& ir, const char* a, const char* b) {
  size_t pos = 0;
  while ((pos = ir.find(a, pos)) != std::string::npos) {
    size_t end = ir.find('\n', pos);
    const std::string line = ir.substr(pos, end == std::string::npos ? std::string::npos
                                                                     : end - pos);
    if (line.find(b) != std::string::npos) return true;
    pos = end;
  }
  return false;
}

const char* kPtx = R"(
.version 8.0
.target sm_80
.address_size 64

.visible .entry probe(
    .param .u64 probe_param_0
)
{
    .reg .b32 %r<3>;
    .reg .b64 %rd<3>;
    .reg .pred %p<1>;
    .shared .align 4 .b32 s0[8];
    .shared .align 8 .b64 s1[4];

    ld.param.u64 %rd1, [probe_param_0];

    cvta.to.shared.u64 %rd2, s0;
    cvta.to.shared.u64 %rd3, s1;

    ld.shared.f32 %r1, [%rd2];
    st.shared.f32 [%rd2], %r1;
    ld.shared.f32 %r1, [s1+16];

    bar.sync 0;

    st.global.f32 [%rd1], %r1;
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
  if (fn.sharedVars.size() != 2 || fn.sharedSize < 64) {
    std::printf("FAIL: bad .shared layout (vars=%zu, size=%d)\n",
                fn.sharedVars.size(), fn.sharedSize);
    return 1;
  }
  std::printf("ok:   .shared layout — s0@%d, s1@%d, scratchpad=%d bytes\n",
              fn.sharedVars[0].offset, fn.sharedVars[1].offset, fn.sharedSize);
  if (fn.sharedVars[0].offset != 0 || fn.sharedVars[1].offset != 32) {
    std::printf("FAIL: expected s0@0 s1@32\n");
    return 1;
  }

  err.clear();
  std::string ir = rougecomp::ptx_to_llvm_ir(*prog, 0, &err);
  if (ir.empty()) {
    std::printf("FAIL: translator error: %s\n", err.c_str());
    return 1;
  }

  // Launch descriptor offset 48 = scratchpad base.
  expect(ir, "getelementptr inbounds i8, ptr %launch, i64 48",
         "scratchpad base read from launch");
  // cvta.to.shared -> add of base + var offset; s1 sits at byte 32.
  if (!expect_line(ir, "= add i64 ", ", 32")) {
    std::printf("FAIL: s1 offset 32 not found\n");
    ++failures;
  } else {
    std::printf("ok:   s1 resolves to scratchpad + 32\n");
  }
  // bar.sync -> runtime barrier.
  expect(ir, "declare void @__rouge_syncthreads(ptr)",
         "runtime barrier declared");
  expect(ir, "call void @__rouge_syncthreads(ptr %launch)",
         "bar.sync lowered to barrier call");
  // ld/st.shared through scratchpad addresses.
  expect(ir, "inttoptr i64 ", "shared register-form addressing (inttoptr)");
  expect(ir, "= bitcast i32 ", "f32 bitcast through shared");
  // block flow after barrier.
  expect(ir, "ret void", "kernel ret");

  std::printf("probe LLVM IR: %zu bytes\n", ir.size());
  if (failures) {
    std::printf("%d structural checks FAILED\n", failures);
    return 1;
  }
  std::printf("all shared/bar structural checks passed\n");
  return 0;
}