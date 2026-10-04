/*
 * Блочный GEMM FP32 + автотюнер конфигурации (rouge-numerics).
 *
 * Здесь три вещи:
 *   1. сборка одного и того же ядра под разные наборы инструкций (базовый,
 *      avx2+fma, avx512f+fma) и выбор варианта по возможностям процессора;
 *   2. разбиение K на части (split_k) с расчётом каждой части своим потоком
 *      и сложением частей;
 *   3. нормализация конфига: мусорные значения не должны приводить к выходу
 *      за границы буферов.
 *
 * Реализация целиком своя: ни BLAS, ни cuBLAS, ни intrinsics — только
 * стандартная библиотека и target-атрибуты компилятора.
 */

#include "rouge/numerics.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "gemm_detail.h"

// ---------------------------------------------------------------------------
// Варианты ISA. Алгоритм один (gemm_kernel.inc), отличаются только целевые
// инструкции и размер регистрового блока микроядра.
// ---------------------------------------------------------------------------

// Базовая ISA процессора (x86-64: SSE2). Всегда доступна, ветка компилятора
// ничего не добавляет, поэтому работает везде.
#define ROUGE_KERNEL_NAME base
#define ROUGE_KERNEL_TARGET
#define ROUGE_KERNEL_MR 4
#define ROUGE_KERNEL_NR 8
#include "gemm_kernel.inc"
#undef ROUGE_KERNEL_TARGET
#undef ROUGE_KERNEL_MR
#undef ROUGE_KERNEL_NR
#undef ROUGE_KERNEL_NAME

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#define ROUGE_NUMERICS_X86 1
#else
#define ROUGE_NUMERICS_X86 0
#endif

#if ROUGE_NUMERICS_X86 && (defined(__GNUC__) || defined(__clang__))
#define ROUGE_HAVE_AVX2 1
#define ROUGE_HAVE_AVX512 1
#else
#define ROUGE_HAVE_AVX2 0
#define ROUGE_HAVE_AVX512 0
#endif

#if ROUGE_HAVE_AVX2
#define ROUGE_KERNEL_NAME avx2
#define ROUGE_KERNEL_TARGET __attribute__((target("avx2,fma")))
#define ROUGE_KERNEL_MR 4
#define ROUGE_KERNEL_NR 16
#include "gemm_kernel.inc"
#undef ROUGE_KERNEL_TARGET
#undef ROUGE_KERNEL_MR
#undef ROUGE_KERNEL_NR
#undef ROUGE_KERNEL_NAME
#endif

#if ROUGE_HAVE_AVX512
#define ROUGE_KERNEL_NAME avx512
#define ROUGE_KERNEL_TARGET __attribute__((target("avx512f,fma")))
#define ROUGE_KERNEL_MR 4
#define ROUGE_KERNEL_NR 32
#include "gemm_kernel.inc"
#undef ROUGE_KERNEL_TARGET
#undef ROUGE_KERNEL_MR
#undef ROUGE_KERNEL_NR
#undef ROUGE_KERNEL_NAME
#endif

namespace {

using rouge_numerics_detail::AlignedBuffer;
using rouge_numerics_detail::GemmCall;

/* Верхняя граница плитки: ограничивает рабочие буферы (2*(tm*tk + tk*tn)
 * float'ов на поток) и не даёт конфигу из внешнего кода съесть память. */
constexpr int kMaxTile = 256;

/* Что умеет этот процессор. Проверка один раз, дальше копится в static. */
enum IsaVariant { kIsaBase, kIsaAvx2, kIsaAvx512 };

bool cpu_has_avx2() {
#if ROUGE_HAVE_AVX2
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
  return false;
#endif
}

bool cpu_has_avx512() {
#if ROUGE_HAVE_AVX512
  return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("fma");
#else
  return false;
#endif
}

/* Автоподбор варианта ISA по процессору. AVX-512 берём только вместе с FMA:
 * голый AVX-512F тут ничего не даёт.
 *
 * Переменная окружения ROUGE_NUMERICS_ISA ("base"/"baseline", "avx2", "avx512")
 * принудительно выбирает вариант — ею пользуются тесты, чтобы проверить
 * ВСЕ собранные пути, а не только тот, что умеет текущая машина. Если
 * процессор не умеет запрошенный вариант, просьба отклоняется: иначе был бы
 * риск исполнения неподдерживаемых инструкций. */
IsaVariant parse_isa_name(const char *name) {
  if (!name)
    return kIsaBase;
  if (std::strcmp(name, "base") == 0 || std::strcmp(name, "baseline") == 0)
    return kIsaBase;
  if (std::strcmp(name, "avx2") == 0)
    return kIsaAvx2;
  if (std::strcmp(name, "avx512") == 0)
    return kIsaAvx512;
  return kIsaBase;
}

IsaVariant detect_isa() {
  if (const char *forced = std::getenv("ROUGE_NUMERICS_ISA")) {
    const IsaVariant want = parse_isa_name(forced);
    const bool ok = (want == kIsaAvx512) ? cpu_has_avx512()
                    : (want == kIsaAvx2) ? cpu_has_avx2()
                                         : true;
    if (ok)
      return want;
  }
  if (cpu_has_avx512())
    return kIsaAvx512;
  if (cpu_has_avx2())
    return kIsaAvx2;
  return kIsaBase;
}

IsaVariant isa_variant() {
  static const IsaVariant v = detect_isa();
  return v;
}

void run_kernel(const GemmCall &call) {
  switch (isa_variant()) {
#if ROUGE_HAVE_AVX512
  case kIsaAvx512:
    run_avx512(call);
    return;
#endif
#if ROUGE_HAVE_AVX2
  case kIsaAvx2:
    run_avx2(call);
    return;
#endif
  default:
    run_base(call);
    return;
  }
}

/* Приводит произвольный конфиг к рабочему: поля <= 0 заменяются значениями по
 * умолчанию, плитки ограничиваются kMaxTile, unroll допускает только 1/2/4,
 * split_k <= 0 превращается в 1 (без разделения K). */
RougeGemmConfig normalize(RougeGemmConfig cfg) {
  const RougeGemmConfig d = rouge_gemm_default_config();
  if (cfg.tile_m <= 0)
    cfg.tile_m = d.tile_m;
  if (cfg.tile_n <= 0)
    cfg.tile_n = d.tile_n;
  if (cfg.tile_k <= 0)
    cfg.tile_k = d.tile_k;
  if (cfg.split_k <= 0)
    cfg.split_k = 1;
  cfg.tile_m = std::min(cfg.tile_m, kMaxTile);
  cfg.tile_n = std::min(cfg.tile_n, kMaxTile);
  cfg.tile_k = std::min(cfg.tile_k, kMaxTile);
  if (cfg.unroll <= 1) {
    cfg.unroll = 1;
  } else if (cfg.unroll >= 4) {
    cfg.unroll = 4;
  } else {
    cfg.unroll = 2;
  }
  return cfg;
}

} // namespace

extern "C" {

RougeGemmConfig rouge_gemm_default_config(void) {
  RougeGemmConfig cfg;
  cfg.tile_m = 64;
  cfg.tile_n = 64;
  cfg.tile_k = 32;
  cfg.unroll = 2;
  cfg.split_k = 1;
  return cfg;
}

void rouge_gemm_f32(int M, int N, int K, const float *A, const float *B, float *C,
                    const RougeGemmConfig *cfg_in) {
  if (M <= 0 || N <= 0 || K <= 0 || !A || !B || !C)
    return;

  const RougeGemmConfig cfg = normalize(cfg_in ? *cfg_in : rouge_gemm_default_config());
  const std::size_t out = static_cast<std::size_t>(M) * static_cast<std::size_t>(N);

  // split_k — верхняя граница числа частей K: пустых частей не делаем, иначе
  // на K < split_k платили бы за потоки, которые не считают ничего.
  const int chunks = std::max(1, std::min(cfg.split_k, K));

  // Границы частей K: максимально равномерно, без дробных шагов.
  std::vector<int> edges(static_cast<std::size_t>(chunks) + 1);
  for (int i = 0; i <= chunks; ++i) {
    edges[static_cast<std::size_t>(i)] = static_cast<int>(static_cast<long long>(K) * i / chunks);
  }

  if (chunks == 1) {
    // Один поток: накопитель — сам C, лишней копии нет.
    AlignedBuffer pa(2 * static_cast<std::size_t>(cfg.tile_m) * cfg.tile_k);
    AlignedBuffer pb(2 * static_cast<std::size_t>(cfg.tile_k) * cfg.tile_n);
    GemmCall call;
    call.m = M;
    call.n = N;
    call.k = K;
    call.a = A;
    call.b = B;
    call.c = C;
    call.tile_m = cfg.tile_m;
    call.tile_n = cfg.tile_n;
    call.tile_k = cfg.tile_k;
    call.unroll = cfg.unroll;
    call.k_begin = 0;
    call.k_end = K;
    call.acc = C;
    call.pack_a[0] = pa.data();
    call.pack_a[1] = pa.data() + static_cast<std::size_t>(cfg.tile_m) * cfg.tile_k;
    call.pack_b[0] = pb.data();
    call.pack_b[1] = pb.data() + static_cast<std::size_t>(cfg.tile_k) * cfg.tile_n;
    rouge_numerics_detail::zero(C, out);
    run_kernel(call);
    return;
  }

  // Несколько частей K: у каждой свой накопитель [M][N] и свои буферы
  // упаковки; в конце части складываются в C.
  AlignedBuffer parts(static_cast<std::size_t>(chunks) * out);
  const std::size_t pa_floats = 2 * static_cast<std::size_t>(cfg.tile_m) * cfg.tile_k;
  const std::size_t pb_floats = 2 * static_cast<std::size_t>(cfg.tile_k) * cfg.tile_n;
  AlignedBuffer packs(static_cast<std::size_t>(chunks) * (pa_floats + pb_floats));

  auto worker = [&](int chunk) {
    float *acc = parts.data() + static_cast<std::size_t>(chunk) * out;
    rouge_numerics_detail::zero(acc, out);
    GemmCall call;
    call.m = M;
    call.n = N;
    call.k = K;
    call.a = A;
    call.b = B;
    call.c = C;
    call.tile_m = cfg.tile_m;
    call.tile_n = cfg.tile_n;
    call.tile_k = cfg.tile_k;
    call.unroll = cfg.unroll;
    call.k_begin = edges[static_cast<std::size_t>(chunk)];
    call.k_end = edges[static_cast<std::size_t>(chunk) + 1];
    call.acc = acc;
    float *base = packs.data() + static_cast<std::size_t>(chunk) * (pa_floats + pb_floats);
    call.pack_a[0] = base;
    call.pack_a[1] = base + static_cast<std::size_t>(cfg.tile_m) * cfg.tile_k;
    call.pack_b[0] = base + pa_floats;
    call.pack_b[1] = base + pa_floats + static_cast<std::size_t>(cfg.tile_k) * cfg.tile_n;
    run_kernel(call);
  };

  // Часть 0 считаем в вызывающем потоке, остальные — на своих.
  std::vector<std::thread> pool;
  pool.reserve(static_cast<std::size_t>(chunks - 1));
  for (int i = 1; i < chunks; ++i)
    pool.emplace_back(worker, i);
  worker(0);
  for (std::thread &t : pool)
    t.join();

  float *sum = parts.data();
  for (int i = 1; i < chunks; ++i) {
    rouge_numerics_detail::add_into(sum, sum, parts.data() + static_cast<std::size_t>(i) * out,
                                    out);
  }
  std::copy(sum, sum + out, C);
}

const char *rouge_gemm_isa_variant(void) {
  switch (isa_variant()) {
  case kIsaAvx512:
    return "avx512+fma";
  case kIsaAvx2:
    return "avx2+fma";
  default:
    return "base+sse2";
  }
}

} // extern "C"