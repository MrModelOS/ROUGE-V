// AOT end-to-end test for prmt.b32 and dp4a/dp2a.
//
// Project invariant: AOT-compiled code and the interpreter agree bit-for-bit.
// The driver runs the NATIVE kernel and the INTERPRETER on the same inputs
// (all seven modes of the canonical prmt_dp4a kernel) and compares the
// resulting words. The arithmetic is integer, so there is no tolerance.
//
// The semantics were pinned against the PTX ISA (9.7.10.7 for prmt,
// 9.7.1.24/9.7.1.25 for dp4a/dp2a) and confirmed on a real NVIDIA card by
// the compiler_nvptx_cuda_gpu_exec test.
//
// ptx2ir signature: define void @prmt_dp4a(i64 out, i32 a, i32 b, i32 mode,
//                                            ptr %launch)
extern "C" void prmt_dp4a(unsigned long long out, unsigned int a,
                          unsigned int b, unsigned int mode, const void* launch);

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "ptx.h"

struct Launch {
  int32_t spec[12] = {0};
  unsigned long long smem = 0;
  void* barrier = nullptr;
};

int main(int argc, char** argv) {
  const std::string ptx_path =
      argc > 1 ? argv[1] : "kernels/prmt_dp4a.ptx";
  FILE* f = std::fopen(ptx_path.c_str(), "rb");
  if (!f) {
    std::printf("SKIP: no %s\n", ptx_path.c_str());
    return 0;
  }
  std::fseek(f, 0, SEEK_END);
  long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::string text((size_t)n, '\0');
  if (std::fread(&text[0], 1, text.size(), f) != text.size()) {
    std::printf("FAIL: short read\n");
    return 1;
  }
  std::fclose(f);

  std::string err;
  auto prog = rouge::parse_ptx(text, &err);
  if (!prog) {
    std::printf("FAIL parse: %s\n", err.c_str());
    return 1;
  }
  int fnIndex = -1;
  for (size_t i = 0; i < prog->functions.size(); ++i)
    if (prog->functions[i].name == "prmt_dp4a") fnIndex = (int)i;
  if (fnIndex < 0) {
    std::printf("FAIL: no prmt_dp4a entry\n");
    return 1;
  }
  const auto& fn = prog->functions[fnIndex];

  // (a, b) per mode: the prmt modes use distinct byte patterns, the dp4a
  // modes use bytes with the high bit set so that the signed/unsigned
  // difference shows up.
  struct Case { unsigned a, b, mode; };
  const Case cases[] = {
      {0x01020304u, 0x0A0B0C0Du, 0},
      {0x01020304u, 0x0A0B0C0Du, 1},
      {0x01020304u, 0x0A0B0C0Du, 2},
      {0x81828384u, 0x8A8B8C8Du, 3},
      {0x81828384u, 0x8A8B8C8Du, 4},
      {0x81828384u, 0x8A8B8C8Du, 5},
      {0x00008384u, 0x00008C8Du, 6},
  };

  int failures = 0, checked = 0;
  for (const Case& c : cases) {
    unsigned native = 0xdeadbeef, interp = 0xdeadbeef;
    Launch launch{};
    prmt_dp4a((unsigned long long)&native, c.a, c.b, c.mode, &launch);

    std::vector<uint8_t> blob((size_t)fn.paramsBlobSize, 0);
    for (const auto& p : fn.params) {
      if (p.index == 0) {
        const unsigned long long dev = (unsigned long long)&interp;
        std::memcpy(blob.data() + p.offset, &dev, 8);
      }
      if (p.index == 1) std::memcpy(blob.data() + p.offset, &c.a, 4);
      if (p.index == 2) std::memcpy(blob.data() + p.offset, &c.b, 4);
      if (p.index == 3) std::memcpy(blob.data() + p.offset, &c.mode, 4);
    }
    if (!rouge::execute_kernel(*prog, fnIndex, blob, 1, 1, 1, 1, 1, 1, &err,
                               nullptr)) {
      std::printf("FAIL interp mode=%u: %s\n", c.mode, err.c_str());
      return 1;
    }
    if (native != interp) {
      std::printf("FAIL mode=%u: native=0x%08x interp=0x%08x\n", c.mode, native,
                  interp);
      ++failures;
    }
    ++checked;
  }
  if (failures == 0)
    std::printf("PASS: AOT prmt/dp4a/dp2a bit-identical to the interpreter "
                "(%d modes)\n",
                checked);
  return failures ? 1 : 0;
}
