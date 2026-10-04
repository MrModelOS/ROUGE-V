// AOT end-to-end test for narrow-float packed conversions (fp8_cvt kernel).
//
// Project invariant: AOT-compiled code and the interpreter agree bit-for-bit.
// The driver calls the NATIVE kernel and the INTERPRETER on identical inputs
// (all 256 byte values x 10 modes, plus float boundaries) and compares.
// The interpreter itself is validated against the OCP specs in
// test_narrow_fp.cpp, so agreement here proves the AOT lowering.

// ptx2ir signature: define void @fp8_cvt(i64 out, i32 in, i32 mode, ptr %launch)
extern "C" void fp8_cvt(unsigned long long out, unsigned int in, unsigned int mode,
                        const void* launch);

#include <cmath>
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

static int failures = 0;

int main(int argc, char** argv) {
  const std::string ptx_path =
      argc > 1 ? argv[1] : "kernels/fp8_cvt.ptx";
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
    if (prog->functions[i].name == "fp8_cvt") fnIndex = (int)i;
  if (fnIndex < 0) {
    std::printf("FAIL: no fp8_cvt entry\n");
    return 1;
  }
  const auto& fn = prog->functions[fnIndex];

  // Param layout from the parse (offsets/widths/index).
  Launch launch{};
  int checked = 0;
  // Byte sweep: every packed-byte input through every decode mode (2,3,6,7,9).
  for (unsigned byte = 0; byte < 256; ++byte) {
    const unsigned packed = (byte << 8) | byte;
    for (unsigned mode : {2u, 3u, 6u, 7u, 9u}) {
      unsigned native = 0xdddddddd;
      fp8_cvt((unsigned long long)&native, packed, mode, &launch);
      // Interpreter on the same input (fill blob by param index).
      std::vector<uint8_t> blob((size_t)fn.paramsBlobSize, 0);
      unsigned interp_out = 0xdddddddd;
      unsigned long long dev = reinterpret_cast<unsigned long long>(&interp_out);
      for (const auto& p : fn.params) {
        if (p.index == 0) std::memcpy(blob.data() + p.offset, &dev, 8);
        if (p.index == 1) std::memcpy(blob.data() + p.offset, &packed, 4);
        if (p.index == 2) std::memcpy(blob.data() + p.offset, &mode, 4);
      }
      if (!rouge::execute_kernel(*prog, fnIndex, blob, 1, 1, 1, 1, 1, 1, &err,
                                 nullptr)) {
        std::printf("FAIL interp mode=%u byte=0x%02x: %s\n", mode, byte, err.c_str());
        return 1;
      }
      if (native != interp_out) {
        if (checked < 5)
          std::printf("FAIL mode=%u byte=0x%02x: native=0x%08x interp=0x%08x\n", mode,
                      byte, native, interp_out);
        ++failures;
      }
      ++checked;
    }
  }
  // Float boundaries through every encode mode (0,1,4,5,8).
  float vals[] = {0.0f, -0.0f, 1.0f, -1.0f, 448.0f, -448.0f, 449.0f, 256.0f,
                  432.0f, 400.0f, 57344.0f, 60000.0f, 28.0f, 7.5f, 6.0f, 4.0f,
                  100.0f, -100.0f, 1e30f, 1e-10f, 0.0625f, 0.125f,
                  std::numeric_limits<float>::quiet_NaN(),
                  std::numeric_limits<float>::infinity(),
                  -std::numeric_limits<float>::infinity()};
  for (float v : vals) {
    unsigned bits = 0;
    std::memcpy(&bits, &v, 4);
    for (unsigned mode : {0u, 1u, 4u, 5u, 8u}) {
      unsigned native = 0xdddddddd;
      fp8_cvt((unsigned long long)&native, bits, mode, &launch);
      unsigned interp_out = 0xdddddddd;
      unsigned long long dev = reinterpret_cast<unsigned long long>(&interp_out);
      std::vector<uint8_t> blob((size_t)fn.paramsBlobSize, 0);
      for (const auto& p : fn.params) {
        if (p.index == 0) std::memcpy(blob.data() + p.offset, &dev, 8);
        if (p.index == 1) std::memcpy(blob.data() + p.offset, &bits, 4);
        if (p.index == 2) std::memcpy(blob.data() + p.offset, &mode, 4);
      }
      if (!rouge::execute_kernel(*prog, fnIndex, blob, 1, 1, 1, 1, 1, 1, &err,
                                 nullptr)) {
        std::printf("FAIL interp mode=%u v=%g: %s\n", mode, (double)v, err.c_str());
        return 1;
      }
      if (native != interp_out) {
        std::printf("FAIL mode=%u v=%g: native=0x%08x interp=0x%08x\n", mode, (double)v,
                    native, interp_out);
        ++failures;
      }
      ++checked;
    }
  }
  if (failures == 0)
    std::printf("PASS: AOT narrow cvt bit-identical to interpreter (%d checks)\n",
                checked);
  return failures ? 1 : 0;
}
