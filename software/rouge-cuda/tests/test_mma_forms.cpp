// Канонические формы mma: uniform-1.0 фрагменты -> каждый D равен ровно K.
//
// mma_forms.ptx содержит по два entry на каждую поддерживаемую форму:
// <form>_ones (все A/B равны 1.0/1, C=0 -> D=K: ручная проверка декода,
// накопления и кодирования) и <form>_lane (AOT-тест сверяет с интерпретатором).
// Здесь проверяются только *_ones против битовых эталонов:
//   f32: 16.0 = 0x41800000 (k16), 8.0 = 0x41000000 (k8)
//   s32: 16 / 8
//   f16: оба халфа 0x4C00 (16.0) / 0x4800 (8.0)

#include "ptx.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int failures = 0;

#define EXPECT_TRUE(cond)                                        \
  do {                                                           \
    if (!(cond)) {                                               \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                \
    }                                                            \
  } while (0)

struct Case {
  const char* entry;
  int nD;             // D-регистров на поток
  uint32_t want[4];   // ожидаемые биты каждого D-регистра
};

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : ".";
  const std::string path = dir + "/mma_forms.ptx";
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) {
    std::printf("SKIP: no %s\n", path.c_str());
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

  // D=K: k16 -> 16 (.0f = 0x41800000, s32 = 16, f16 = 0x4C00 оба халфа),
  // k8 -> 8 (0x41000000, 8, 0x4800).
  const uint32_t f32_16 = 0x41800000u, f32_8 = 0x41000000u;
  const uint32_t f16_16 = 0x4C004C00u, f16_8 = 0x48004800u;
  const Case cases[] = {
      {"mma_bf16k16f32_ones", 4, {f32_16, f32_16, f32_16, f32_16}},
      {"mma_e4m3k16f32_ones", 4, {f32_16, f32_16, f32_16, f32_16}},
      {"mma_e5m2k16f32_ones", 4, {f32_16, f32_16, f32_16, f32_16}},
      {"mma_u8k16s32_ones", 4, {16, 16, 16, 16}},
      {"mma_s8k16s32_ones", 4, {16, 16, 16, 16}},
      {"mma_f16k16f16_ones", 2, {f16_16, f16_16}},
      {"mma_f16k8f32_ones", 4, {f32_8, f32_8, f32_8, f32_8}},
      {"mma_bf16k8f16_ones", 2, {f16_8, f16_8}},
  };

  for (const auto& cs : cases) {
    int fnIndex = -1;
    for (size_t i = 0; i < prog->functions.size(); ++i)
      if (prog->functions[i].name == cs.entry) fnIndex = (int)i;
    EXPECT_TRUE(fnIndex >= 0);
    if (fnIndex < 0) continue;

    std::vector<uint32_t> out(32 * 4, 0xdeadbeefu);
    unsigned long long dev = reinterpret_cast<unsigned long long>(out.data());
    std::vector<uint8_t> blob(8, 0);
    std::memcpy(blob.data(), &dev, 8);
    if (!rouge::execute_kernel(*prog, fnIndex, blob, 1, 1, 1, 32, 1, 1, &err,
                               nullptr)) {
      std::printf("FAIL exec %s: %s\n", cs.entry, err.c_str());
      ++failures;
      continue;
    }
    int bad = 0;
    for (int lane = 0; lane < 32; ++lane)
      for (int s = 0; s < cs.nD; ++s) {
        const uint32_t got = out[(size_t)lane * (size_t)cs.nD + (size_t)s];
        if (got != cs.want[s]) {
          if (bad < 4)
            std::printf("%s lane=%d s=%d: got=0x%08x want=0x%08x\n", cs.entry,
                        lane, s, got, cs.want[s]);
          ++bad;
        }
      }
    EXPECT_TRUE(bad == 0);
  }

  if (failures == 0)
    std::printf("PASS: mma forms ones (8 forms, D==K bit-exact)\n");
  return failures ? 1 : 0;
}
