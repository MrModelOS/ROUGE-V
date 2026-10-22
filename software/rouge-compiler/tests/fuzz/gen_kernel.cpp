// ROUGE-V: генератор случайных PTX-ядер для дифференциального фазера.
//
// Идея: у нас уже есть три независимых пути исполнения (текстовый
// интерпретатор PTX, AOT-низнинг в LLVM IR, настоящий GPU) и есть
// инфраструктура побитового сравнения. Не хватает только ГЕНЕРАТОРА:
// пятнадцать канонических ядер руками не покрывают краевые случаи
// (satfinite, невыровненный сдвиг, инвертированный предикат, знак при
// делении на отрицательное).
//
// Генератор выдаёт ядро фиксированной формы:
//
//   fuzz_kernel(int* out, const int* in, unsigned n, unsigned seed)
//
//   сетка 1 блок x n потоков; поток i обрабатывает элемент i и пишет
//   K = 4 результата в out[i*K + j].
//
// Тело — случайная цепочка операций над четырьмя регистрами. Набор
// операций выбран не случайно: это ровно те места, где семантика PTX
// неочевидна и где компиляторы чаще всего ошибаются:
//
//   * shr/shl с ПЕРЕМЕННОЙ величиной сдвига (PTX маскирует её по ширине
//     типа: 5 бит для 32-битного регистра);
//   * prmt с произвольным селектором (включая msb=1 — репликация знака);
//   * dp4a.s32/dp4a.u32 (разница знаковости байт);
//   * setp + предицированная операция (@p add.s32);
//   * div/rem с отрицательными операндами (знак результата);
//   * popc/clz/bfi/bfe (границы по битам);
//   * min/max со знаковыми и беззнаковыми операндами.
//
// Ничего «сложного» (shared memory, атомики, барьеры) здесь намеренно
// нет: это первый слой фазера, а не весь фазер.
//
// Использование:
//   gen_kernel --seed N --out kernel.ptx --input in.bin --meta meta.txt

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// Простой детерминированный ГПСЧ (splitmix64): одинаковый seed -> один и
// тот же kernel на любой машине.
struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 0x1234567ull) {}
  uint64_t next() {
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D5CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  unsigned u32(unsigned n) { return n ? (unsigned)(next() % n) : 0; }
  int i32() { return (int)(uint32_t)next(); }
};

constexpr int kOutPerThread = 4;
constexpr int kN = 64;  // элементов на ядро

// Список операций. Формат: текст шаблона с %1..%4 (регистры) и %5
// (непосредственный операнд).
enum class Op {
  Add, Sub, Mul, Mad, Shl, ShrU, ShrS, And, Or, Xor, Not, MinS, MaxS,
  Popc, Clz, Bfi, Bfe, Prmt, Dp4aS, Dp4aU, Setp, Div, Rem,
};

// Имена операций (в том же порядке, что и enum Op) — для режима --ops.
const char* const opNames[] = {
    "add", "sub", "mul", "mad", "shl", "shru", "shrs", "and", "or", "xor",
    "not", "mins", "maxs", "popc", "clz", "bfi", "bfe", "prmt", "dp4as",
    "dp4au", "setp", "div", "rem"};

const char* opName(Op op) {
  switch (op) {
    case Op::Add: return "add.s32";
    case Op::Sub: return "sub.s32";
    case Op::Mul: return "mul.lo.s32";
    case Op::Mad: return "mad.lo.s32";
    case Op::Shl: return "shl.b32";
    case Op::ShrU: return "shr.u32";
    case Op::ShrS: return "shr.s32";
    case Op::And: return "and.b32";
    case Op::Or: return "or.b32";
    case Op::Xor: return "xor.b32";
    case Op::Not: return "not.b32";
    case Op::MinS: return "min.s32";
    case Op::MaxS: return "max.s32";
    case Op::Popc: return "popc.b32";
    case Op::Clz: return "clz.b32";
    case Op::Bfi: return "bfi.b32";
    case Op::Bfe: return "bfe.b32";
    case Op::Prmt: return "prmt.b32";
    case Op::Dp4aS: return "dp4a.s32.s32";
    case Op::Dp4aU: return "dp4a.u32.u32";
    case Op::Setp: return "setp.lt.s32";
    case Op::Div: return "div.s32";
    case Op::Rem: return "rem.s32";
  }
  return "add.s32";
}

// Операции с двумя регистрами-источниками, остальные — с одним.
bool isBinary(Op op) {
  switch (op) {
    case Op::Add: case Op::Sub: case Op::Mul: case Op::Mad: case Op::Shl:
    case Op::ShrU: case Op::ShrS: case Op::And: case Op::Or: case Op::Xor:
    case Op::MinS: case Op::MaxS: case Op::Bfi: case Op::Bfe: case Op::Prmt:
    case Op::Dp4aS: case Op::Dp4aU: case Op::Setp: case Op::Div: case Op::Rem:
      return true;
    default:
      return false;
  }
}

// У операций prmt/bfi/bfe третий операнд — непосредственное число,
// выбираемое так, чтобы попадать в краевые случаи (msb=1 у prmt).
unsigned immediateFor(Op op, Rng& rng) {
  switch (op) {
    case Op::Prmt: {
      // Селектор prmt: 4 поля по 4 бита. Ставим и обычные значения, и
      // значения с msb=1 (репликация знака) — редкий случай, который
      // компиляторы забывают.
      const unsigned mode = rng.u32(3);
      if (mode == 0) return rng.u32(16);              // 0..15: только байты
      if (mode == 1) return 0x8000u | rng.u32(16);    // msb у старшего поля
      return 0xF000u | rng.u32(16);                   // все поля с msb
    }
    default:
      return (unsigned)rng.i32();
  }
}

// The "phases" template: shared memory + a barrier. A different kernel
// SHAPE (two phases over the threads), not a different arithmetic chain, so
// it is generated separately from the chain template:
//
//   __shared__ int sdata[64];
//   sdata[tid] = in[tid] + seed;
//   __syncthreads();
//   out[tid*K + j] = f(sdata[n - 1 - tid], j, seed);
//
// The REVERSE index in the second phase is deliberate: it is not
// "thread + const", so the pass must keep it scalar. The point of this
// template is the semantics of the barrier (the phases) and of the shared
// memory (a per-block scratchpad), not the vectoriser.
std::string phasesKernel(uint64_t seed) {
  Rng rng(seed);
  const unsigned k1 = 1 + rng.u32(7);
  const unsigned k2 = 1 + rng.u32(101);
  std::string p;
  p += "// ROUGE-V differential fuzzer kernel (generated, do not edit).\n";
  p += "// template=phases seed=" + std::to_string(seed) + "\n";
  p += ".version 8.0\n.target sm_75\n.address_size 64\n\n";
  p += ".visible .entry fuzz_kernel(\n";
  p += "    .param .u64 fuzz_kernel_param_0,\n";
  p += "    .param .u64 fuzz_kernel_param_1,\n";
  p += "    .param .u32 fuzz_kernel_param_2,\n";
  p += "    .param .u32 fuzz_kernel_param_3\n";
  p += ")\n{\n";
  p += "    .shared .align 4 .b8 sdata[256];\n";
  p += "    .reg .b32 %r<16>;\n";
  p += "    .reg .b64 %rd<16>;\n\n";
  p += "    ld.param.u64 %rd1, [fuzz_kernel_param_0];\n";
  p += "    ld.param.u64 %rd2, [fuzz_kernel_param_1];\n";
  p += "    ld.param.u32 %r1, [fuzz_kernel_param_2];\n";
  p += "    ld.param.u32 %r2, [fuzz_kernel_param_3];\n";
  p += "    cvta.to.global.u64 %rd3, %rd1;\n";
  p += "    cvta.to.global.u64 %rd4, %rd2;\n";
  p += "    mov.u32 %r6, %tid.x;\n";
  p += "    mul.wide.u32 %rd7, %r6, 4;\n";
  // The first phase: the shared memory is filled by the thread with the
  // same index.
  p += "    add.s64 %rd8, %rd4, %rd7;\n";
  p += "    ld.global.s32 %r7, [%rd8];\n";
  p += "    add.s32 %r7, %r7, %r2;\n";
  p += "    cvta.shared.u64 %rd9, sdata;\n";
  p += "    add.s64 %rd10, %rd9, %rd7;\n";
  p += "    st.shared.s32 [%rd10], %r7;\n";
  p += "    bar.sync 0;\n";
  // The second phase: a DIFFERENT thread reads every element (n-1-tid).
  p += "    sub.s32 %r8, %r1, %r6;\n";
  p += "    sub.s32 %r8, %r8, 1;\n";
  p += "    mul.wide.u32 %rd11, %r8, 4;\n";
  p += "    add.s64 %rd12, %rd9, %rd11;\n";
  p += "    ld.shared.s32 %r9, [%rd12];\n";
  p += "    mad.lo.s32 %r10, %r9, " + std::to_string(k1) + ", " +
         std::to_string(k2) + ";\n";
  p += "    xor.b32 %r11, %r10, %r2;\n";
  p += "    mul.lo.s32 %r12, %r10, 3;\n";
  p += "    sub.s32 %r13, %r11, %r12;\n";
  const char* vals[4] = {"%r10", "%r11", "%r12", "%r13"};
  for (int j = 0; j < kOutPerThread; ++j) {
    p += "    mul.lo.s32 %r14, %r6, " +
         std::to_string(kOutPerThread) + ";\n";
    p += "    add.s32 %r14, %r14, " + std::to_string(j) + ";\n";
    p += "    mul.wide.u32 %rd13, %r14, 4;\n";
    p += "    add.s64 %rd13, %rd13, %rd3;\n";
    p += "    st.global.s32 [%rd13], " + std::string(vals[j]) + ";\n";
  }
  p += "    ret;\n}\n";
  return p;
}

// The "demoted" template: the same shape as "phases", but the shared-memory
// addresses are built the way nvcc actually emits them — a 32-bit
// "mov.u32 %r, <shared symbol>" base, chained "add.s32", and "[reg+imm]"
// accesses — instead of "cvta.shared.u64" + "add.s64". This exercises the
// front-end's demoted-shared address recovery.
std::string demotedKernel(uint64_t seed) {
  Rng rng(seed);
  const unsigned k1 = 1 + rng.u32(7);
  const unsigned k2 = 1 + rng.u32(101);
  std::string p;
  p += "// ROUGE-V differential fuzzer kernel (generated, do not edit).\n";
  p += "// template=demoted seed=" + std::to_string(seed) + "\n";
  p += ".version 8.0\n.target sm_75\n.address_size 64\n\n";
  p += ".visible .entry fuzz_kernel(\n";
  p += "    .param .u64 fuzz_kernel_param_0,\n";
  p += "    .param .u64 fuzz_kernel_param_1,\n";
  p += "    .param .u32 fuzz_kernel_param_2,\n";
  p += "    .param .u32 fuzz_kernel_param_3\n";
  p += ")\n{\n";
  p += "    .shared .align 4 .b8 sdata[256];\n";
  p += "    .reg .b32 %r<32>;\n";
  p += "    .reg .b64 %rd<16>;\n\n";
  p += "    ld.param.u64 %rd1, [fuzz_kernel_param_0];\n";
  p += "    ld.param.u64 %rd2, [fuzz_kernel_param_1];\n";
  p += "    ld.param.u32 %r1, [fuzz_kernel_param_2];\n";
  p += "    ld.param.u32 %r2, [fuzz_kernel_param_3];\n";
  p += "    cvta.to.global.u64 %rd3, %rd1;\n";
  p += "    cvta.to.global.u64 %rd4, %rd2;\n";
  p += "    mov.u32 %r6, %tid.x;\n";
  p += "    mul.wide.u32 %rd7, %r6, 4;\n";
  // The first phase: the shared memory is filled by the thread with the
  // same index (demoted shared addressing).
  p += "    add.s64 %rd8, %rd4, %rd7;\n";
  p += "    ld.global.s32 %r7, [%rd8];\n";
  p += "    add.s32 %r7, %r7, %r2;\n";
  p += "    mov.u32 %r20, sdata;\n";
  p += "    mul.lo.s32 %r21, %r6, 4;\n";
  p += "    add.s32 %r22, %r20, %r21;\n";
  p += "    st.shared.s32 [%r22], %r7;\n";
  p += "    bar.sync 0;\n";
  // The second phase: a DIFFERENT thread reads every element (n-1-tid),
  // with one access using the "[reg+imm]" form.
  p += "    sub.s32 %r8, %r1, %r6;\n";
  p += "    sub.s32 %r8, %r8, 1;\n";
  p += "    mul.lo.s32 %r23, %r8, 4;\n";
  p += "    add.s32 %r24, %r20, %r23;\n";
  p += "    ld.shared.s32 %r9, [%r24];\n";
  p += "    mad.lo.s32 %r10, %r9, " + std::to_string(k1) + ", " +
       std::to_string(k2) + ";\n";
  p += "    xor.b32 %r11, %r10, %r2;\n";
  p += "    mul.lo.s32 %r12, %r10, 3;\n";
  p += "    sub.s32 %r13, %r11, %r12;\n";
  const char* vals[4] = {"%r10", "%r11", "%r12", "%r13"};
  for (int j = 0; j < kOutPerThread; ++j) {
    p += "    mul.lo.s32 %r14, %r6, " +
         std::to_string(kOutPerThread) + ";\n";
    p += "    add.s32 %r14, %r14, " + std::to_string(j) + ";\n";
    p += "    mul.wide.u32 %rd13, %r14, 4;\n";
    p += "    add.s64 %rd13, %rd13, %rd3;\n";
    p += "    st.global.s32 [%rd13], " + std::string(vals[j]) + ";\n";
  }
  p += "    ret;\n}\n";
  return p;
}

// The three files of one fuzz case: the PTX text, the input blob and the
// metadata the driver reads.
int writeOutputs(const std::string& ptx, const std::vector<int>& input,
                 uint64_t seed, const char* outPath, const char* inPath,
                 const char* metaPath) {
  FILE* f = fopen(outPath, "wb");
  if (!f) {
    std::fprintf(stderr, "gen_kernel: cannot write %s\n", outPath);
    return 1;
  }
  std::fwrite(ptx.data(), 1, ptx.size(), f);
  fclose(f);

  f = fopen(inPath, "wb");
  if (!f) {
    std::fprintf(stderr, "gen_kernel: cannot write %s\n", inPath);
    return 1;
  }
  std::fwrite(input.data(), sizeof(int), input.size(), f);
  fclose(f);

  f = fopen(metaPath, "wb");
  if (!f) {
    std::fprintf(stderr, "gen_kernel: cannot write %s\n", metaPath);
    return 1;
  }
  std::fprintf(f, "n=%d\nk=%d\nseed=%llu\n", kN, kOutPerThread,
               (unsigned long long)seed);
  fclose(f);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  uint64_t seed = 1;
  const char* outPath = nullptr;
  const char* inPath = nullptr;
  const char* metaPath = nullptr;
  // --ops "add,prmt,dp4a" — задать цепочку операций вручную (через запятую).
  // Это режим «одна операция за раз»: он локализует расхождение до конкретной
  // инструкции, а не до случайной цепочки.
  std::string forcedOps;
  // --template chain|phases — the kernel SHAPE. "chain" (the default) is
  // the per-thread arithmetic chain; "phases" is shared memory + a barrier.
  std::string tpl = "chain";
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--seed") && i + 1 < argc) {
      seed = strtoull(argv[++i], nullptr, 10);
    } else if (!strcmp(argv[i], "--out") && i + 1 < argc) {
      outPath = argv[++i];
    } else if (!strcmp(argv[i], "--input") && i + 1 < argc) {
      inPath = argv[++i];
    } else if (!strcmp(argv[i], "--meta") && i + 1 < argc) {
      metaPath = argv[++i];
    } else if (!strcmp(argv[i], "--ops") && i + 1 < argc) {
      forcedOps = argv[++i];
    } else if (!strcmp(argv[i], "--template") && i + 1 < argc) {
      tpl = argv[++i];
      if (tpl != "chain" && tpl != "phases" && tpl != "demoted") {
        std::fprintf(stderr, "gen_kernel: unknown template '%s'\n",
                     tpl.c_str());
        return 2;
      }
    } else {
      std::fprintf(stderr,
                   "usage: %s --seed N --out k.ptx --input in.bin --meta m.txt\n",
                   argv[0]);
      return 2;
    }
  }
  if (!outPath || !inPath || !metaPath) {
    std::fprintf(stderr, "gen_kernel: missing --out/--input/--meta\n");
    return 2;
  }

  Rng rng(seed);

  // Входные данные: псевдослучайные int32 (в том числе отрицательные и
  // экстремальные — для них особенно важны div/rem и знаковые сравнения).
  std::vector<int> input((size_t)kN);
  for (int i = 0; i < kN; ++i) {
    uint32_t r = (uint32_t)rng.next();
    // Раз в 8 элементов — extreme value (0x80000000, 0x7fffffff, -1).
    if ((i & 7) == 0) {
      static const int extremes[4] = {0x7fffffff, (int)0x80000000, -1, 0};
      input[(size_t)i] = extremes[rng.u32(4)];
    } else {
      input[(size_t)i] = (int)r;
    }
  }

  // Шаблон "phases": другая ФОРМА ядра (общая память + барьер), поэтому
  // генератор цепочки операций здесь не участвует.
  if (tpl == "phases") return writeOutputs(phasesKernel(seed), input, seed,
                                           outPath, inPath, metaPath);
  if (tpl == "demoted") return writeOutputs(demotedKernel(seed), input, seed,
                                            outPath, inPath, metaPath);

  // Цепочка операций: одна на каждый «результат» (K регистров), плюс
  // несколько общих шагов.
  std::vector<Op> ops((size_t)6);
  for (int i = 0; i < 6; ++i) {
    // Смесь: часть операций бинарные, часть унарные.
    static const Op pool[] = {
        Op::Add,  Op::Sub,  Op::Mul,   Op::Mad,   Op::Shl,  Op::ShrU,
        Op::ShrS, Op::And,  Op::Or,    Op::Xor,   Op::Not,  Op::MinS,
        Op::MaxS, Op::Popc, Op::Clz,   Op::Bfi,   Op::Bfe,  Op::Prmt,
        Op::Dp4aS, Op::Dp4aU, Op::Setp, Op::Div, Op::Rem};
    ops[(size_t)i] = pool[rng.u32(sizeof(pool) / sizeof(pool[0]))];
  }
  // Режим «ровно эти операции»: цепочка задаётся извне.
  if (!forcedOps.empty()) {
    ops.clear();
    size_t pos = 0;
    while (pos < forcedOps.size()) {
      size_t comma = forcedOps.find(',', pos);
      std::string name =
          forcedOps.substr(pos, comma == std::string::npos ? std::string::npos
                                                           : comma - pos);
      Op op = Op::Add;
      bool known = true;
      for (int k = 0; k < 24; ++k) {
        if (name == opNames[k]) { op = (Op)k; known = true; break; }
      }
      if (!known) {
        std::fprintf(stderr, "gen_kernel: unknown op '%s'\n", name.c_str());
        return 2;
      }
      ops.push_back(op);
      if (comma == std::string::npos) break;
      pos = comma + 1;
    }
  }
  // В принудительном режиме длина цепочки = число заданных операций.
  const int kSteps = forcedOps.empty() ? 6 : (int)ops.size();
  // Гарантируем разнообразие: в цепочке есть и бинарная, и предицированная
  // операция (иначе ветка предиката не проверяется).
  bool hasSetp = false, hasBinary = false;
  for (Op op : ops) {
    if (op == Op::Setp) hasSetp = true;
    if (isBinary(op)) hasBinary = true;
  }
  if (forcedOps.empty()) {
    if (!hasSetp) ops[(size_t)rng.u32((unsigned)kSteps)] = Op::Setp;
    if (!hasBinary) ops[(size_t)rng.u32((unsigned)kSteps)] = Op::Add;
  }

  // Текст ядра.
  std::string ptx;
  ptx += "// ROUGE-V differential fuzzer kernel (generated, do not edit).\n";
  ptx += "// seed=" + std::to_string(seed) + "\n";
  ptx += ".version 8.0\n.target sm_75\n.address_size 64\n\n";
  ptx += ".visible .entry fuzz_kernel(\n";
  ptx += "    .param .u64 fuzz_kernel_param_0,\n";
  ptx += "    .param .u64 fuzz_kernel_param_1,\n";
  ptx += "    .param .u32 fuzz_kernel_param_2,\n";
  ptx += "    .param .u32 fuzz_kernel_param_3\n";
  ptx += ")\n{\n";
  ptx += "    .reg .b32 %r<16>;\n";
  ptx += "    .reg .b64 %rd<8>;\n";
  ptx += "    .reg .f32 %f<4>;\n";
  ptx += "    .reg .pred %p<4>;\n\n";
  ptx += "    ld.param.u64 %rd1, [fuzz_kernel_param_0];\n";
  ptx += "    ld.param.u64 %rd2, [fuzz_kernel_param_1];\n";
  ptx += "    ld.param.u32 %r1, [fuzz_kernel_param_2];\n";
  ptx += "    ld.param.u32 %r2, [fuzz_kernel_param_3];\n";
  ptx += "    cvta.to.global.u64 %rd3, %rd1;\n";
  ptx += "    cvta.to.global.u64 %rd4, %rd2;\n";
  ptx += "    mov.u32 %r3, %ctaid.x;\n";
  ptx += "    mov.u32 %r4, %ntid.x;\n";
  ptx += "    mul.wide.u32 %rd5, %r3, %r4;\n";
  ptx += "    mov.u32 %r6, %tid.x;\n";
  ptx += "    cvt.u64.u32 %rd6, %r6;\n";
  ptx += "    add.s64 %rd5, %rd5, %rd6;\n";
  ptx += "    cvt.u32.u64 %r5, %rd5;\n";
  ptx += "    setp.ge.u32 %p1, %r5, %r1;\n";
  ptx += "    @%p1 bra $L_out;\n";
  // Вход: in + i*4 (в 64-битном адресе).
  ptx += "    mul.wide.u32 %rd7, %r5, 4;\n";
  ptx += "    add.s64 %rd4, %rd4, %rd7;\n";
  ptx += "    ld.global.s32 %r7, [%rd4];\n";
  ptx += "    mov.u32 %r8, %r2;\n";  // зерно в цепочке
  // Все четыре выходных регистра инициализируются: неинициализированный
  // регистр в PTX — неопределённое значение, и сравнение с железом было бы
  // сравнением мусора (а не проверкой операции).
  ptx += "    mov.s32 %r9, %r7;\n";
  ptx += "    mov.s32 %r10, %r7;\n";
  ptx += "    mov.s32 %r11, %r7;\n";
  ptx += "    mov.s32 %r12, %r7;\n";
  // %r13 тоже используется как операнд — он обязан быть инициализирован:
  // неинициализированный регистр в PTX — неопределённое значение (на
  // железе там мусор, в интерпретаторе — ноль), и сравнение с железом
  // тогда проверяет не операцию, а мусор.
  ptx += "    mov.u32 %r13, %r2;\n";
  ptx += "\n";

  // Регистры-«результаты»: %r9..%r12. Промежуточные: %r13..%r15.
  const char* dstReg[4] = {"%r9", "%r10", "%r11", "%r12"};
  for (int step = 0; step < kSteps; ++step) {
    Op op = ops[(size_t)step];
    const char* d = dstReg[step % 4];
    if (op == Op::Setp) {
      // setp + предицированное сложение: проверяем и предикат, и то, что
      // предицированная запись НЕ меняет регистр, когда предикат ложен.
      const char* a = "%r7";
      const char* b = "%r8";
      ptx += "    setp.lt.s32 %p2, " + std::string(a) + ", " + std::string(b) + ";\n";
      ptx += "    mov.s32 " + std::string(d) + ", " + std::string(a) + ";\n";
      ptx += "    @%p2 add.s32 " + std::string(d) + ", " + std::string(d) + ", " +
             std::string(b) + ";\n";
      continue;
    }
    if (isBinary(op)) {
      const char* a = (rng.u32(2) == 0) ? "%r7" : "%r8";
      const char* b = (rng.u32(3) == 0) ? "%r7" : ((rng.u32(2) == 0) ? "%r8" : "%r13");
      const unsigned imm = immediateFor(op, rng);
      if (op == Op::Prmt) {
        ptx += "    prmt.b32 " + std::string(d) + ", " + std::string(a) + ", " +
               std::string(b) + ", " + std::to_string(imm) + ";\n";
      } else if (op == Op::Dp4aS || op == Op::Dp4aU) {
        ptx += "    " + std::string(opName(op)) + " " + std::string(d) + ", " +
               std::string(a) + ", " + std::string(b) + ", 0;\n";
      } else if (op == Op::Bfi) {
        // bfi.type f, a, b, c, d — the position and the length are SEPARATE
        // operands (PTX ISA 9.7.9). The old generator packed them into one
        // (pos | len << 8), which is not PTX at all.
        const unsigned pos = rng.u32(33);  // 0..32, including "past the msb"
        const unsigned len = rng.u32(33);  // 0..32, including "the whole word"
        ptx += "    bfi.b32 " + std::string(d) + ", " + std::string(a) + ", " +
               std::string(b) + ", " + std::to_string(pos) + ", " +
               std::to_string(len) + ";\n";
      } else if (op == Op::Bfe) {
        // bfe.type d, a, b, c — there is no separate "second source": b is
        // the position and c the length.
        const bool sign = rng.u32(2) != 0;
        const unsigned pos = rng.u32(33);
        const unsigned len = rng.u32(33);
        ptx += "    " + std::string(sign ? "bfe.s32" : "bfe.u32") + " " +
               std::string(d) + ", " + std::string(a) + ", " +
               std::to_string(pos) + ", " + std::to_string(len) + ";\n";
      } else if (op == Op::Mad) {
        // mad: d = a*b + c — третий операнд обязателен.
        ptx += "    mad.lo.s32 " + std::string(d) + ", " + std::string(a) + ", " +
               std::string(b) + ", " + std::string(a) + ";\n";
      } else if (op == Op::Div || op == Op::Rem) {
        // Делитель не должен быть нулём: деление на ноль в PTX — неопределённый
        // случай (железо даёт -1, интерпретатор — 0), и сравнивать его
        // бессмысленно. Поэтому берём (x | 1).
        ptx += "    or.b32 %r15, " + std::string(b) + ", 1;\n";
        ptx += "    " + std::string(opName(op)) + " " + std::string(d) + ", " +
               std::string(a) + ", %r15;\n";
      } else {
        ptx += "    " + std::string(opName(op)) + " " + std::string(d) + ", " +
               std::string(a) + ", " + std::string(b) + ";\n";
      }
      continue;
    }
    // Унарная операция.
    const char* a = (rng.u32(2) == 0) ? "%r7" : "%r8";
    ptx += "    " + std::string(opName(op)) + " " + std::string(d) + ", " +
           std::string(a) + ";\n";
  }

  // Запись результатов: out[(i*K + j)*4].
  // Арифметика индекса — в 32-битном регистре: "add.u32" на
  // 64-битном регистре — невалидный PTX (на железе результат
  // неопределённый, и сравнение ловило бы мусор).
  for (int j = 0; j < kOutPerThread; ++j) {
    ptx += "    mul.lo.s32 %r14, %r5, " + std::to_string(kOutPerThread) + ";\n";
    ptx += "    add.s32 %r14, %r14, " + std::to_string(j) + ";\n";
    ptx += "    mul.wide.u32 %rd7, %r14, 4;\n";
    ptx += "    add.s64 %rd7, %rd7, %rd3;\n";
    ptx += "    st.global.s32 [%rd7], " + std::string(dstReg[j]) + ";\n";
  }
  ptx += "$L_out:\n";
  ptx += "    ret;\n";
  ptx += "}\n";

  return writeOutputs(ptx, input, seed, outPath, inPath, metaPath);
}
