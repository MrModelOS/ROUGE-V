// AOT end-to-end test for the bit-field instructions: bfi, bfe, prmt.
//
// Project invariant: the AOT-compiled code and the interpreter agree
// bit-for-bit. That invariant is necessary but NOT sufficient here: both
// paths used to read a packed selector and to insert b into a instead of a
// into b, so they agreed perfectly on a non-PTX answer. The test therefore
// compares three things:
//
//   1. the native code (ptx2ir + clang, linked into this binary),
//   2. our PTX interpreter (the reference for the fuzzer),
//   3. the PTX ISA's pseudo-code, re-implemented below independently of
//      both (bfi/bfe: ISA 9.7.9; prmt: 9.7.10).
//
// Any disagreement is a failure. The cases include the clamps (pos past the
// msb, len == 0, len > the word), the sign extension of bfe, and the sign
// replication of prmt's selector nibbles.
//
// ptx2ir signature: define void @bitfield(i64 out, i64 in, i32 n, ptr %launch)

extern "C" void bitfield(unsigned long long out, unsigned long long in,
                         unsigned int n, const void* launch);

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

namespace {

// ---- the ISA's pseudo-code, written out independently --------------------

// bfi.type f, a, b, c, d: pos = c & 0xff, len = d & 0xff; f = b; bits of a
// are copied into f at pos, up to the msb of the word.
uint32_t isa_bfi(uint32_t a, uint32_t b, unsigned pos, unsigned len) {
  pos &= 0xffu;
  len &= 0xffu;
  uint32_t f = b;
  for (unsigned i = 0; i < len && pos + i <= 31; ++i) {
    const uint32_t bit = (a >> i) & 1u;
    const unsigned at = pos + i;
    f = (f & ~(1u << at)) | (bit << at);
  }
  return f;
}

// bfe.type d, a, b, c: pos = b & 0xff, len = c & 0xff. len == 0 gives 0.
uint32_t isa_bfe(uint32_t a, unsigned pos, unsigned len, bool sign) {
  pos &= 0xffu;
  len &= 0xffu;
  if (len == 0) return 0;
  const unsigned top = (pos + len - 1u <= 31) ? pos + len - 1u : 31u;
  const uint32_t sbit = sign ? ((a >> top) & 1u) : 0u;
  uint32_t d = 0;
  for (unsigned i = 0; i < 32; ++i) {
    uint32_t bit;
    if (i < len && pos + i <= 31)
      bit = (a >> (pos + i)) & 1u;
    else
      bit = sbit;
    d |= bit << i;
  }
  return d;
}

// prmt.b32 d, a, b, c, mode 0: {a,b} is the 64-bit source; nibble i of c
// selects the byte for output byte i; bit 3 of a nibble replicates the sign
// bit of the selected byte instead of the byte itself.
uint32_t isa_prmt(uint32_t a, uint32_t b, uint32_t sel) {
  const uint64_t src = (static_cast<uint64_t>(b) << 32) | a;
  uint32_t out = 0;
  for (unsigned i = 0; i < 4; ++i) {
    const unsigned nib = (sel >> (4 * i)) & 0xfu;
    const unsigned idx = nib & 7u;
    const unsigned byte = static_cast<unsigned>((src >> (8 * idx)) & 0xffu);
    const unsigned r =
        (nib & 8u) ? ((byte & 0x80u) ? 0xffu : 0x00u) : byte;
    out |= static_cast<uint32_t>(r) << (8 * i);
  }
  return out;
}

// The nine cases of the kernel, in the kernel's order.
void isa_case(uint32_t x, uint32_t y, uint32_t out[9]) {
  out[0] = isa_bfi(x, y, 3, 7);
  out[1] = isa_bfi(x, y, 0, 32);
  out[2] = isa_bfi(x, y, 30, 8);
  out[3] = isa_bfi(x, y, 40, 4);
  out[4] = isa_bfe(x, 5, 11, /*sign=*/false);
  out[5] = isa_bfe(x, 3, 12, /*sign=*/true);
  out[6] = isa_bfe(x, 0, 0, /*sign=*/false);
  out[7] = isa_prmt(x, y, 0x7140u);
  out[8] = isa_prmt(x, y, 0x8B7Au);
}

}  // namespace

int main(int argc, char** argv) {
  const std::string ptx_path =
      argc > 1 ? argv[1] : "kernels/bitfield.ptx";
  FILE* f = std::fopen(ptx_path.c_str(), "rb");
  if (!f) {
    std::printf("SKIP: no %s\n", ptx_path.c_str());
    return 0;
  }
  std::fseek(f, 0, SEEK_END);
  const long size = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::string text(static_cast<size_t>(size), '\0');
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
    if (prog->functions[i].name == "bitfield") fnIndex = static_cast<int>(i);
  if (fnIndex < 0) {
    std::printf("FAIL: no bitfield entry\n");
    return 1;
  }
  const auto& fn = prog->functions[fnIndex];

  // Values chosen so that every clamp and both signs appear: an all-ones
  // word, an all-zeros word, only the top bit set, and a spread.
  const unsigned n = 16;
  std::vector<uint32_t> input = {
      0x00000000u, 0xffffffffu, 0x80000000u, 0x7fffffffu,
      0x81828384u, 0x01020304u, 0xdeadbeefu, 0x5a5a5a5au,
      0x00000001u, 0xffff0000u, 0x0000ffffu, 0x12345678u,
      0x89abcdefu, 0x0f0f0f0fu, 0xf0f0f0f0u, 0x00ff00ffu,
  };
  input.resize(n, 0x13579bdfu);
  std::vector<uint32_t> expected(n * 9, 0);
  std::vector<uint32_t> interp(n * 9, 0xdeadbeefu);
  std::vector<uint32_t> native(n * 9, 0xdeadbeefu);

  for (unsigned i = 0; i < n; ++i)
    isa_case(input[i], input[i] ^ 0x5a5a5a5au, &expected[i * 9]);

  // 1) the native code. The kernel reads the thread id from the launch
  // struct, so it is invoked once per thread, the way the runtime would.
  for (unsigned i = 0; i < n; ++i) {
    Launch launch{};
    launch.spec[0] = static_cast<int32_t>(i);   // tidx
    launch.spec[6] = static_cast<int32_t>(n);   // ntidx
    bitfield(reinterpret_cast<unsigned long long>(native.data()),
             reinterpret_cast<unsigned long long>(input.data()), n, &launch);
  }

  // 2) the interpreter.
  {
    std::vector<uint8_t> blob(static_cast<size_t>(fn.paramsBlobSize), 0);
    for (const auto& p : fn.params) {
      if (p.index == 0) {
        const unsigned long long v =
            reinterpret_cast<unsigned long long>(interp.data());
        std::memcpy(blob.data() + p.offset, &v, 8);
      } else if (p.index == 1) {
        const unsigned long long v =
            reinterpret_cast<unsigned long long>(input.data());
        std::memcpy(blob.data() + p.offset, &v, 8);
      } else if (p.index == 2) {
        std::memcpy(blob.data() + p.offset, &n, 4);
      }
    }
    if (!rouge::execute_kernel(*prog, fnIndex, blob, 1, 1, 1, n, 1, 1, &err,
                               nullptr)) {
      std::printf("FAIL interp: %s\n", err.c_str());
      return 1;
    }
  }

  // 3) the comparison.
  int failures = 0;
  static const char* names[9] = {"bfi(3,7)", "bfi(0,32)", "bfi(30,8)",
                                 "bfi(40,4)", "bfe.u32(5,11)", "bfe.s32(3,12)",
                                 "bfe.u32(0,0)", "prmt(0x7140)",
                                 "prmt(0x8B7A)"};
  for (unsigned i = 0; i < n; ++i) {
    for (unsigned j = 0; j < 9; ++j) {
      const unsigned k = i * 9 + j;
      if (native[k] != expected[k] || interp[k] != expected[k]) {
        std::printf("FAIL x=0x%08x %-14s isa=0x%08x native=0x%08x interp=0x%08x\n",
                    input[i], names[j], expected[k], native[k], interp[k]);
        ++failures;
      }
    }
  }

  if (failures == 0)
    std::printf("PASS: bfi/bfe/prmt match the PTX ISA in both paths "
                "(%u words, 9 cases each)\n",
                n);
  return failures ? 1 : 0;
}
