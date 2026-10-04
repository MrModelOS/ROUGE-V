// Driver-путь occupancy: CUfunction -> KernelStats -> калькулятор.
//
// Калькулятор покрыт test_occupancy.cpp, сравнение с настоящим драйвером —
// wave1_occupancy.cu. Здесь проверяется клей между ними: разрешение хэндлов,
// WithFlags, blockSizeLimit и callback динамической shared.

#include <rouge/cuda.h>

#include <cstdio>

// То же ядро, что occmid в wave1_occupancy.cu (shared 1K): числа обязаны
// совпасть с теми, что печатает программа на настоящем стеке.
static const char* kPtx = R"(
.version 9.4
.target sm_75
.address_size 64
.visible .entry occmid(
    .param .u64 occmid_param_0,
    .param .u64 occmid_param_1,
    .param .u32 occmid_param_2
)
{
    .reg .pred %p<2>;
    .reg .f32 %f<8>;
    .reg .b32 %r<12>;
    .reg .b64 %rd<6>;
    .shared .align 4 .b8 smem[1024];
    ld.param.u64 %rd1, [occmid_param_0];
    ld.param.u64 %rd2, [occmid_param_1];
    ld.param.u32 %r1, [occmid_param_2];
    mov.u32 %r2, %tid.x;
    cvta.to.shared.u64 %rd3, smem;
    cvt.u64.u32 %rd4, %r2;
    shl.b64 %rd4, %rd4, 2;
    add.s64 %rd4, %rd3, %rd4;
    mov.f32 %f1, 0f3F800000;
    st.shared.f32 [%rd4], %f1;
    bar.sync 0;
    ld.shared.f32 %f2, [%rd3];
    cvta.to.global.u64 %rd5, %rd2;
    st.global.f32 [%rd5], %f2;
    ret;
}
)";

static int failures = 0;

#define CHECK_OK(expr)                                                          \
  do {                                                                          \
    const CUresult _r = (expr);                                                 \
    if (_r != CUDA_SUCCESS) {                                                   \
      std::printf("FAIL %s:%d: %s -> %d\n", __FILE__, __LINE__, #expr, (int)_r); \
      return 1;                                                                 \
    }                                                                           \
  } while (0)

#define EXPECT_EQ(got, want)                                                  \
  do {                                                                        \
    const long long _g = (long long)(got);                                    \
    const long long _w = (long long)(want);                                   \
    if (_g != _w) {                                                           \
      std::printf("FAIL %s:%d: %s = %lld, want %lld\n", __FILE__, __LINE__,   \
                  #got, _g, _w);                                              \
      ++failures;                                                             \
    }                                                                         \
  } while (0)

static size_t dyn_for_block(int blockSize) { return (size_t)blockSize * 16; }

int main() {
  CHECK_OK(cuInit(0));
  CUcontext ctx = nullptr;
  CHECK_OK(cuCtxCreate(&ctx, 0, 0));
  CUmodule mod = nullptr;
  CHECK_OK(cuModuleLoadData(&mod, kPtx));
  CUfunction fn = nullptr;
  CHECK_OK(cuModuleGetFunction(&fn, mod, "occmid"));

  // Атрибуты через driver API.
  int shared = -1, regs = -1, maxT = -1;
  CHECK_OK(cuFuncGetAttribute(&shared, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fn));
  EXPECT_EQ(shared, 1024);
  CHECK_OK(cuFuncGetAttribute(&maxT, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, fn));
  EXPECT_EQ(maxT, 1024);
  CHECK_OK(cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, fn));
  if (regs <= 0) {
    std::printf("FAIL numRegs = %d, want > 0\n", regs);
    ++failures;
  } else {
    std::printf("info: numRegs bound = %d\n", regs);
  }

  int nb = -1;
  CHECK_OK(cuOccupancyMaxActiveBlocksPerMultiprocessor(&nb, fn, 256, 0));
  if (nb <= 0 || nb > 16) {
    std::printf("FAIL active(256) = %d, want 1..16\n", nb);
    ++failures;
  } else {
    std::printf("info: active(256) = %d\n", nb);
  }
  // Невозможный размер — успех с нулём.
  nb = -1;
  CHECK_OK(cuOccupancyMaxActiveBlocksPerMultiprocessor(&nb, fn, 2048, 0));
  EXPECT_EQ(nb, 0);
  // Чужие флаги — отказ.
  nb = -1;
  if (cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&nb, fn, 256, 0, 0xdeadbeef) !=
      CUDA_ERROR_INVALID_VALUE) {
    std::printf("FAIL bad flags accepted\n");
    ++failures;
  }
  // Флаги 0 и 1 — один ответ.
  int nb0 = -1, nb1 = -1;
  CHECK_OK(cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&nb0, fn, 256, 0, 0));
  CHECK_OK(cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&nb1, fn, 256, 0, 1));
  EXPECT_EQ(nb0, nb1);

  // Potential без лимита и с лимитом + callback динамики.
  int mg = -1, bs = -1;
  CHECK_OK(cuOccupancyMaxPotentialBlockSize(&mg, &bs, fn, nullptr, 0, 0));
  if (bs <= 0 || bs > 1024 || bs % 32 != 0 || mg <= 0) {
    std::printf("FAIL potential = (%d, %d)\n", mg, bs);
    ++failures;
  } else {
    std::printf("info: potential = (minGrid=%d, block=%d)\n", mg, bs);
  }
  int mg2 = -1, bs2 = -1;
  CHECK_OK(cuOccupancyMaxPotentialBlockSize(&mg2, &bs2, fn, dyn_for_block, 0, 128));
  if (bs2 <= 0 || bs2 > 128) {
    std::printf("FAIL potential-limited = (%d, %d)\n", mg2, bs2);
    ++failures;
  } else {
    std::printf("info: potential-limited = (minGrid=%d, block=%d)\n", mg2, bs2);
  }

  // Невалидный хэндл — отказ, а не падение.
  nb = -1;
  if (cuOccupancyMaxActiveBlocksPerMultiprocessor(&nb, nullptr, 256, 0) !=
      CUDA_ERROR_INVALID_VALUE) {
    std::printf("FAIL null func accepted\n");
    ++failures;
  }

  CHECK_OK(cuModuleUnload(mod));
  CHECK_OK(cuCtxDestroy(ctx));
  if (failures == 0) std::printf("PASS: driver occupancy path\n");
  return failures ? 1 : 0;
}
