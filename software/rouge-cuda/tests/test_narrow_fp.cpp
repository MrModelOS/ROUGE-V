// Исчерпывающая проверка narrow float форматов (e4m3/e5m2/e3m2/e2m3/e2m1).
//
// Векторы — из открытых спек, а не по памяти:
//   OFP8 Table 1-2: e4m3 (bias 7, max 448=0x7E, NaN только 0x7F/0xFF, без Inf),
//                   e5m2 (bias 15, max 57344=0x7B, Inf 0x7C/0xFC);
//   OCP MX v1.0 Table 7: e3m2 (bias 3, max 28), e2m3 (bias 1, max 7.5),
//                         e2m1 (bias 1, max 6).
// Проверяется через настоящие cvt-инструкции интерпретатора:
// decode — все значения каждого формата; encode — границы, RNE-ties,
// saturation, NaN-правила PTX (NaN->NaN для e4m3/e5m2, NaN->+max для FP6/FP4).

#include "ptx.h"

#include <cmath>
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

// Ядро: %r1 = входные биты, тело делает cvt и складывает u32-результат в %r2.
static uint32_t run_cvt(const char* body, uint32_t in) {
  char ptx[4096];
  std::snprintf(ptx, sizeof ptx,
                ".version 9.4\n.target sm_90\n.address_size 64\n"
                ".visible .entry k(.param .u64 p0, .param .u32 p1)\n{\n"
                " .reg .b32 %%r<8>;\n .reg .b64 %%rd<3>;\n .reg .pred %%p<2>;\n"
                " ld.param.u64 %%rd1, [p0];\n"
                " ld.param.u32 %%r1, [p1];\n"
                " cvta.to.global.u64 %%rd1, %%rd1;\n%s\n"
                " st.global.u32 [%%rd1], %%r2;\n"
                " ret;\n}\n",
                body);
  std::string err;
  auto prog = rouge::parse_ptx(ptx, &err);
  if (!prog) {
    std::printf("FAIL parse: %s\nbody: %s\n", err.c_str(), body);
    ++failures;
    return 0xdeadbeef;
  }
  void* out = std::calloc(64, 1);
  unsigned long long dev = reinterpret_cast<unsigned long long>(out);
  std::vector<uint8_t> blob(16, 0);
  std::memcpy(blob.data(), &dev, 8);
  std::memcpy(blob.data() + 8, &in, 4);
  if (!rouge::execute_kernel(*prog, 0, blob, 1, 1, 1, 1, 1, 1, &err, nullptr)) {
    std::printf("FAIL exec %s: %s\n", body, err.c_str());
    ++failures;
    std::free(out);
    return 0xdeadbeef;
  }
  uint32_t v = 0;
  std::memcpy(&v, out, 4);
  std::free(out);
  return v;
}

static uint32_t f2u(float f) {
  uint32_t u = 0;
  std::memcpy(&u, &f, 4);
  return u;
}
static float u2f(uint32_t u) {
  float f = 0;
  std::memcpy(&f, &u, 4);
  return f;
}

// f32 -> packed x2 (то же значение в обоих слотах).
static uint32_t encode2(const char* fmt, float v) {
  char body[256];
  std::snprintf(body, sizeof body, " cvt.rn.satfinite.%s.f32 %%r2, %%r1, %%r1;\n", fmt);
  return run_cvt(body, f2u(v));
}

// packed x2 -> f16x2 -> f32 младшего слота. Байт кладётся в оба слота,
// возвращается младший (оба одинаковы — неоднозначности нет).
static float decode_lo(const char* fmt, uint8_t byte) {
  char body[256];
  std::snprintf(body, sizeof body,
                " cvt.rn.f16x2.%s %%r3, %%r1;\n"
                " cvt.f32.f16 %%r2, %%r3;\n",
                fmt);
  // вход %r1: байт в обоих слотах b16
  const uint32_t packed = (uint32_t)byte | ((uint32_t)byte << 8);
  return u2f(run_cvt(body, packed));
}

int main() {
  // --- e4m3 точечные векторы из спеки ---
  EXPECT_TRUE(decode_lo("e4m3x2", 0x00) == 0.0f);
  EXPECT_TRUE(std::signbit(decode_lo("e4m3x2", 0x80)));
  EXPECT_TRUE(decode_lo("e4m3x2", 0x80) == 0.0f);
  EXPECT_TRUE(decode_lo("e4m3x2", 0x7E) == 448.0f);
  EXPECT_TRUE(std::isnan(decode_lo("e4m3x2", 0x7F)));
  EXPECT_TRUE(std::isnan(decode_lo("e4m3x2", 0xFF)));
  // subnormals: 0x01 -> 2^-9, 0x07 -> 2^-6*0.875, 0x08 -> 2^-6, 0x38 -> 1.0
  EXPECT_TRUE(decode_lo("e4m3x2", 0x01) == std::ldexp(1.0f, -9));
  EXPECT_TRUE(decode_lo("e4m3x2", 0x07) == std::ldexp(0.875f, -6));
  EXPECT_TRUE(decode_lo("e4m3x2", 0x08) == std::ldexp(1.0f, -6));
  EXPECT_TRUE(decode_lo("e4m3x2", 0x38) == 1.0f);

  // --- e5m2 точечные ---
  EXPECT_TRUE(decode_lo("e5m2x2", 0x00) == 0.0f);
  EXPECT_TRUE(decode_lo("e5m2x2", 0x7B) == 57344.0f);
  EXPECT_TRUE(decode_lo("e5m2x2", 0x7C) == std::numeric_limits<float>::infinity());
  EXPECT_TRUE(std::isnan(decode_lo("e5m2x2", 0x7D)));
  EXPECT_TRUE(decode_lo("e5m2x2", 0x01) == std::ldexp(1.0f, -16));
  EXPECT_TRUE(decode_lo("e5m2x2", 0x3C) == 1.0f);

  // --- e3m2 / e2m3 / e2m1 точечные (контейнеры посчитаны вручную) ---
  // e3m2 (S EEE MM): +28=0x1F, -28=0x3F, +0.25=0x04, submax 0.1875=0x03, submin 0.0625=0x01
  EXPECT_TRUE(decode_lo("e3m2x2", 0x00) == 0.0f);
  EXPECT_TRUE(decode_lo("e3m2x2", 0x1F) == 28.0f);
  EXPECT_TRUE(decode_lo("e3m2x2", 0x3F) == -28.0f);
  EXPECT_TRUE(decode_lo("e3m2x2", 0x04) == 0.25f);
  EXPECT_TRUE(decode_lo("e3m2x2", 0x03) == 0.1875f);
  EXPECT_TRUE(decode_lo("e3m2x2", 0x01) == 0.0625f);
  EXPECT_TRUE(decode_lo("e3m2x2", 0x0C) == 1.0f);
  // e2m3 (S EE MMM): +7.5=0x1F, -7.5=0x3F, +1.0=0x08, submax 0.875=0x07, submin 0.125=0x01
  EXPECT_TRUE(decode_lo("e2m3x2", 0x00) == 0.0f);
  EXPECT_TRUE(decode_lo("e2m3x2", 0x1F) == 7.5f);
  EXPECT_TRUE(decode_lo("e2m3x2", 0x3F) == -7.5f);
  EXPECT_TRUE(decode_lo("e2m3x2", 0x08) == 1.0f);
  EXPECT_TRUE(decode_lo("e2m3x2", 0x07) == 0.875f);
  EXPECT_TRUE(decode_lo("e2m3x2", 0x01) == 0.125f);
  // e2m1 (S EE M в центральных битах [5:2]): поле->значение/контейнер:
  // 0->0/0x00, 1->.5/0x04, 2->1/0x08, 3->1.5/0x0C,
  // 4->2/0x10, 5->3/0x14, 6->4/0x18, 7->6/0x1C (+0x20 на контейнер за знак)
  EXPECT_TRUE(decode_lo("e2m1x2", 0x00) == 0.0f);
  EXPECT_TRUE(decode_lo("e2m1x2", 0x04) == 0.5f);
  EXPECT_TRUE(decode_lo("e2m1x2", 0x08) == 1.0f);
  EXPECT_TRUE(decode_lo("e2m1x2", 0x0C) == 1.5f);
  EXPECT_TRUE(decode_lo("e2m1x2", 0x10) == 2.0f);
  EXPECT_TRUE(decode_lo("e2m1x2", 0x14) == 3.0f);
  EXPECT_TRUE(decode_lo("e2m1x2", 0x18) == 4.0f);
  EXPECT_TRUE(decode_lo("e2m1x2", 0x1C) == 6.0f);
  EXPECT_TRUE(decode_lo("e2m1x2", 0x3C) == -6.0f);

  // --- encode: saturation ---
  EXPECT_TRUE((encode2("e4m3x2", 449.0f) & 0xff) == 0x7E);
  EXPECT_TRUE((encode2("e4m3x2", 1e30f) & 0xff) == 0x7E);
  EXPECT_TRUE((encode2("e4m3x2", -449.0f) & 0xff) == 0xFE);
  EXPECT_TRUE((encode2("e4m3x2", std::numeric_limits<float>::infinity()) & 0xff) == 0x7E);
  EXPECT_TRUE((encode2("e4m3x2", std::numeric_limits<float>::quiet_NaN()) & 0xff) == 0x7F);
  EXPECT_TRUE((encode2("e5m2x2", 60000.0f) & 0xff) == 0x7B);
  EXPECT_TRUE((encode2("e5m2x2", std::numeric_limits<float>::infinity()) & 0xff) == 0x7B);
  EXPECT_TRUE((encode2("e5m2x2", std::numeric_limits<float>::quiet_NaN()) & 0xff) == 0x7F);
  EXPECT_TRUE((encode2("e3m2x2", 100.0f) & 0x3f) == 0x3F);
  EXPECT_TRUE(encode2("e3m2x2", -100.0f) == 0x3F3F);  // -28: знак уже внутри поля
  EXPECT_TRUE((encode2("e3m2x2", std::numeric_limits<float>::quiet_NaN()) & 0x3f) == 0x3F);
  EXPECT_TRUE((encode2("e2m3x2", 100.0f) & 0x3f) == 0x3F);
  EXPECT_TRUE(encode2("e2m3x2", -100.0f) == 0x3F3F);  // -7.5
  EXPECT_TRUE((encode2("e2m3x2", std::numeric_limits<float>::quiet_NaN()) & 0x3f) == 0x3F);
  EXPECT_TRUE(encode2("e2m1x2", 100.0f) == 0x3C3C);  // saturate в max-поле 0xF
  EXPECT_TRUE(encode2("e2m1x2", 6.0f) == 0x1C1C);
  EXPECT_TRUE(encode2("e2m1x2", 4.0f) == 0x1818);
  EXPECT_TRUE(encode2("e2m1x2", -6.0f) == 0x3C3C);

  // --- encode: RNE ties ---
  // e4m3: 432 ровно посередине 384 (0x7C) и 448 (0x7E) -> чётный 0x7E.
  EXPECT_TRUE((encode2("e4m3x2", 432.0f) & 0xff) == 0x7E);
  // 400 ровно посередине 384 (0x7C, LSB 0) и 416 (0x7D, LSB 1) -> 0x7C.
  EXPECT_TRUE((encode2("e4m3x2", 400.0f) & 0xff) == 0x7C);

  // --- exhaustive roundtrip ---
  // Каждый слот независимо: decode байта, затем encode обратно. Ожидание —
  // тот же байт, кроме документированных особенностей:
  //  - e5m2 Inf (0x7C/0xFC): satfinite даёт max (0x7B/0xFB), не Inf;
  //  - e5m2 NaN: OFP8 не определяет пэйлоад — наш канонический NaN
  //    знак-сохраняющий (0x7F/0xFF);
  //  - e4m3 NaN (0x7F/0xFF): сохраняются; Inf-паттернов нет;
  //  - FP6/FP4: NaN-паттернов нет, всё roundtrip'ится;
  //  - e2m1 живёт в центральных 4 битах контейнера.
  // Слоты сравниваются по отдельности (младший и старший байт enc).
  for (int fmt = 0; fmt < 5; ++fmt) {
    const char* names[] = {"e4m3x2", "e5m2x2", "e3m2x2", "e2m3x2", "e2m1x2"};
    const int widths[] = {8, 8, 6, 6, 4};
    const int W = widths[fmt];
    const int N = 1 << W;
    for (int v = 0; v < N; ++v) {
      // decode байта, затем encode обратно. Слоты независимы: ожидаем тот же
      // байт в обоих слотах enc. Особенности:
      //  - e5m2 Inf (0x7C/0xFC): satfinite даёт max (0x7B/0xFB), не Inf;
      //  - e5m2 NaN: OFP8 не определяет пэйлоад — наш канонический NaN
      //    знак-сохраняющий (0x7F/0xFF);
      //  - e4m3 NaN (0x7F/0xFF): сохраняются; Inf-паттернов нет;
      //  - FP6/FP4: NaN-паттернов нет, всё roundtrip'ится.
      // e2m1 живёт в центральных 4 битах контейнера.
      const uint8_t byte = (W == 4) ? (uint8_t)(v << 2) : (uint8_t)v;
      float f = decode_lo(names[fmt], byte);
      const uint32_t enc = encode2(names[fmt], f);
      const uint32_t lo = enc & 0xFF, hi = (enc >> 8) & 0xFF;
      bool isNan = std::isnan(f);
      bool isInf = std::isinf(f);
      uint32_t want = byte;
      if (fmt == 1 && isNan) want = (byte & 0x80) ? 0xFF : 0x7F;
      else if (fmt == 1 && isInf) want = (byte & 0x80) ? 0xFB : 0x7B;
      if (lo != want || hi != want) {
        std::printf("FAIL roundtrip %s 0x%02x -> %g -> 0x%02x%02x\n", names[fmt], byte, f,
                    hi, lo);
        ++failures;
      }
    }
  }

  if (failures == 0) std::printf("PASS: narrow float formats exhaustive\n");
  return failures ? 1 : 0;
}
