// Юнит-тест occupancy-калькулятора: числа сверены с настоящим драйвером.
//
// Ground truth снят пробой на MX450 (sm_75), ядро occ_k: shared=1024,
// regs=34, maxT=1024:
//   bs=128 -> 8, bs=256 -> 4, bs=512 -> 2, bs=1024 -> 1,
//   potential -> (minGrid=14, block=1024) при 14 SM,
//   availDyn(4 blocks, 256 threads) = 15360.
// Краевые случаи тоже с пробы: bs=2048 -> 0, dyn=1M -> 0.
//
// Здесь калькулятор вызывается напрямую (без API-слоя) — API-путь проверяет
// wave1_occupancy.cu через сравнение с настоящим стеком.

#include <cstdio>

#include "occupancy.h"

static int failures = 0;

#define EXPECT_EQ(got, want)                                                        \
  do {                                                                              \
    const long long _g = (long long)(got);                                          \
    const long long _w = (long long)(want);                                         \
    if (_g != _w) {                                                                  \
      std::printf("FAIL %s:%d: %s = %lld, want %lld\n", __FILE__, __LINE__, #got,    \
                  _g, _w);                                                          \
      ++failures;                                                                   \
    }                                                                               \
  } while (0)

int main() {
  // occ_k: regs=34 (настоящие, из пробы), static=1024.
  rouge::KernelStats occ{34, 1024, 0, 0, 75};
  EXPECT_EQ(rouge::occ_active_blocks(occ, 128, 0), 8);
  EXPECT_EQ(rouge::occ_active_blocks(occ, 256, 0), 4);
  EXPECT_EQ(rouge::occ_active_blocks(occ, 512, 0), 2);
  EXPECT_EQ(rouge::occ_active_blocks(occ, 1024, 0), 1);
  EXPECT_EQ(rouge::occ_active_blocks(occ, 2048, 0), 0);
  EXPECT_EQ(rouge::occ_active_blocks(occ, 256, 1u << 20), 0);
  EXPECT_EQ(rouge::occ_active_blocks(occ, 0, 0), 0);
  EXPECT_EQ(rouge::occ_active_blocks(occ, -16, 0), 0);

  // ek: regs=8, shared=0.
  rouge::KernelStats ek{8, 0, 0, 0, 75};
  EXPECT_EQ(rouge::occ_active_blocks(ek, 256, 0), 4);
  EXPECT_EQ(rouge::occ_active_blocks(ek, 1024, 0), 1);
  EXPECT_EQ(rouge::occ_active_blocks(ek, 128, 0), 8);

  // Potential: перебор сверху, early-out на полном SM.
  int mg = -1, bs = -1;
  rouge::occ_potential_block_size(&mg, &bs, occ, nullptr, 0, /*numSms=*/14);
  EXPECT_EQ(mg, 14);
  EXPECT_EQ(bs, 1024);
  mg = -1;
  bs = -1;
  rouge::occ_potential_block_size(&mg, &bs, occ, nullptr, 0, /*numSms=*/1);
  EXPECT_EQ(mg, 1);
  EXPECT_EQ(bs, 1024);

  // Available dynamic shared.
  size_t avail = 123;
  if (!rouge::occ_available_dynamic_smem(&avail, occ, 4, 256) || avail != 15360) {
    std::printf("FAIL avail(4,256) = %zu, want 15360\n", avail);
    ++failures;
  }
  avail = 123;
  if (rouge::occ_available_dynamic_smem(&avail, occ, 0, 256)) {
    std::printf("FAIL avail(0,256) should refuse\n");
    ++failures;
  } else if (avail != 123) {
    std::printf("FAIL avail(0,256) touched output\n");
    ++failures;
  }
  // Невозможная конфигурация (столько блоков не влезет) — отказ.
  avail = 123;
  if (rouge::occ_available_dynamic_smem(&avail, occ, 16, 256)) {
    std::printf("FAIL avail(16,256) should refuse (only 4 fit)\n");
    ++failures;
  }

  // kernel_stats: max-live оценка по dataflow (ручной подсчёт ниже).
  const char* ptx =
      ".version 9.4\n.target sm_75\n.address_size 64\n"
      ".visible .entry k(.param .u64 p0)\n{\n"
      " .reg .b32 %r<8>;\n"
      " .reg .b64 %rd<4>;\n"
      " .shared .align 4 .b32 s[64];\n"
      " ld.param.u64 %rd1, [p0];\n"      // def rd1(2)
      " mov.u32 %r1, %tid.x;\n"          // def r1; live: {rd1}=2
      " add.s64 %rd2, %rd1, 8;\n"        // use rd1 def rd2; live: {rd1(2),r1}=3
      " mul.lo.s32 %r2, %r1, 4;\n"       // use r1 def r2; live: {rd2(2),r1}=3
      " add.s32 %r3, %r2, %r1;\n"        // use r2,r1 def r3; live: {rd2(2),r2,r1}=4 <- max
      " st.global.u32 [%rd2], %r3;\n"    // use rd2,r3; live: {rd2(2),r3}=3
      " ret;\n}\n";
  std::string err;
  auto prog = rouge::parse_ptx(ptx, &err);
  if (!prog) {
    std::printf("FAIL parse: %s\n", err.c_str());
    ++failures;
  } else {
    rouge::KernelStats st;
    if (!rouge::kernel_stats(*prog, 0, &st)) {
      std::printf("FAIL kernel_stats refused\n");
      ++failures;
    } else {
      // Max-live = 4 (см. разбор выше). Сумма деклараций дала бы 8+8=16 —
      // оценка обязана быть точнее декларации, иначе это откат, а не анализ.
      EXPECT_EQ(st.numRegs, 4);
      EXPECT_EQ((long long)st.staticSmem, 256);
      EXPECT_EQ(st.targetSm, 75);
    }
    if (rouge::kernel_stats(*prog, 7, &st)) {
      std::printf("FAIL kernel_stats accepted a bad index\n");
      ++failures;
    }
    // else: индекс вне диапазона — отказ, а не падение.
  }

  if (failures == 0)
    std::printf("PASS: occupancy calculator matches the probed driver numbers\n");
  return failures ? 1 : 0;
}
