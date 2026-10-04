// См. occupancy.h: документированный алгоритм из открытого cuda_occupancy.h
// для референсного sm_75. Здесь только арифметика; разрешение хэндлов
// (stub/CUfunction -> модуль) живёт в runtime_api.cpp и driver_api.cpp.

#include "occupancy.h"

#include "rouge_internal.h"

#include <algorithm>
#include <climits>
#include <string>
#include <unordered_map>
#include <vector>

namespace rouge {
namespace {

// sm_75: архитектурные пределы (все сверены пробой с настоящим драйвером).
constexpr int kWarpSize = 32;
constexpr int kMaxThreadsPerBlock = 1024;
constexpr int kMaxThreadsPerSm = 1024;       // Turing уполовинил против Volta!
constexpr int kMaxBlocksPerSm = 16;          // Turing (minor 5)
constexpr int kRegsPerSm = 65536;
constexpr int kRegsPerBlock = 65536;
constexpr int kRegAllocGranularity = 256;    // на варп
constexpr int kMaxRegsPerThread = 256;       // таблица заголовка для major 7
constexpr int kSubPartitions = 4;
constexpr size_t kSmemAllocGranularity = 256;
constexpr size_t kSmemPerSmDefault = 65536;  // default carveout
constexpr size_t kSmemPerBlockLimit = 49152;  // default per-block limit

int divUp(int x, int y) { return (x + (y - 1)) / y; }
int roundUp(int x, int y) { return y * divUp(x, y); }

// Turing buckets: 32K и 64K (cudaOccAlignUpShmemSizeVoltaPlus, ветка isTuring).
// Возвращает false, если объём не влезает ни в один bucket.
bool alignUpTuring(size_t* size) {
  if (*size <= 32u * 1024u) *size = 32u * 1024u;
  else if (*size <= 64u * 1024u) *size = 64u * 1024u;
  else return false;
  return true;
}

// ---- оценка живых регистров (max-live dataflow) ------------------------------
// Предикаты (.pred, 1 бит) не считаются вовсе: numRegs у драйвера — это GPR.
// 64-битный регистр весит 2, остальные GPR — 1.

int reg_weight(const PtxFunction& fn, const std::string& name) {
  const auto it = fn.regBits.find(name);
  if (it == fn.regBits.end()) return 0;  // не регистр: символ, литерал, спецрегистр
  if (it->second <= 1) return 0;          // .pred — отдельный файл
  return (it->second >= 64) ? 2 : 1;
}

// Базовое имя регистра из операнда: "[rd2+8]" -> "rd2", "r10|p1" -> "r10",
// "[sdata]" -> "sdata" (символ, вес 0 через reg_weight).
std::string reg_base(const std::string& tok) {
  std::string t = tok;
  if (t.size() >= 2 && t.front() == '[' && t.back() == ']')
    t = t.substr(1, t.size() - 2);
  const size_t bar = t.find('|');
  if (bar != std::string::npos) t = t.substr(0, bar);
  const size_t pm = t.find_first_of("+-");
  if (pm != std::string::npos && pm > 0) t = t.substr(0, pm);
  if (!t.empty() && t[0] == '%') t = t.substr(1);
  return t;
}

// Безопасная верхняя граница: сумма .reg-деклараций (откат, если dataflow
// неприменим). Завышает, поэтому occupancy с ней — только занижен.
int declared_regs_bound(const PtxFunction& fn) {
  int n = 0;
  for (const auto& kv : fn.regBits) {
    if (kv.second <= 1) continue;
    n += (kv.second >= 64) ? 2 : 1;
  }
  return n;
}

// Классический backward dataflow: liveIn[i] = use[i] ∪ (liveOut[i] − def[i]).
// Возвращает максимум веса живых GPR на любой программной точке, или -1 когда
// анализ неприменим (call в теле — межпроцедурного анализа нет).
int max_live_regs(const PtxFunction& fn) {
  const size_t n = fn.body.size();
  if (n == 0) return -1;
  // Номера регистрам раздаём по факту упоминания.
  std::unordered_map<std::string, int> ids;
  std::vector<int> weights;
  const auto id_of = [&](const std::string& tok) -> int {
    const std::string base = reg_base(tok);
    if (base.empty()) return -1;
    const int w = reg_weight(fn, base);
    if (w == 0) return -1;
    const auto it = ids.find(base);
    if (it != ids.end()) return it->second;
    const int id = static_cast<int>(weights.size());
    ids[base] = id;
    weights.push_back(w);
    return id;
  };

  std::vector<std::vector<int>> defs(n), uses(n);
  std::vector<std::vector<int>> succ(n);
  for (size_t i = 0; i < n; ++i) {
    const auto& ins = fn.body[i];
    const std::string& op = ins.op;
    if (op.rfind("call", 0) == 0) return -1;  // межпроцедурного анализа нет
    if (op == "ret" || op == "exit") continue;  // нет def/use/succ
    if (op == "bra" || op == "bra.uni") {
      // Цель — use нет, но переход есть. Без предиката — без fallthrough.
      const auto it = fn.labels.find(ins.args.empty() ? "" : ins.args[0]);
      if (it != fn.labels.end()) succ[i].push_back(it->second);
      if (!ins.pred.empty() && i + 1 < n) succ[i].push_back(static_cast<int>(i + 1));
      continue;
    }
    if (op == "bar.sync" || op == "bar.warp.sync" || op == "barrier.sync" ||
        op.rfind("membar.", 0) == 0 || op.rfind("fence.", 0) == 0) {
      if (i + 1 < n) succ[i].push_back(static_cast<int>(i + 1));
      continue;
    }
    // ld/st/atomic: A[0] — адрес (use) для st/atom, приёмник (def) для ld/atom.
    const bool is_store = op.rfind("st.", 0) == 0;
    const bool is_atom =
        op.rfind("atom.", 0) == 0 || op.rfind("red.", 0) == 0;
    if (is_store || is_atom) {
      for (const auto& a : ins.args) {
        const int id = id_of(a);
        if (id >= 0) (is_store || &a != &ins.args[0] ? uses[i] : defs[i]).push_back(id);
      }
      // atom с приёмником: A[0] — def, а не use (цикл выше записал его в uses).
      if (is_atom && !ins.args.empty()) {
        const int id = id_of(ins.args[0]);
        if (id >= 0) {
          auto& u = uses[i];
          u.erase(std::remove(u.begin(), u.end(), id), u.end());
          defs[i].push_back(id);
        }
      }
      if (i + 1 < n) succ[i].push_back(static_cast<int>(i + 1));
      continue;
    }
    // Обычная форма: A[0] — приёмник (кроме setp/vote/shfl-предикатов, которые
    // пишут в pred-файл и здесь игнорируются), остальное — источники.
    if (!ins.args.empty() && op.rfind("setp.", 0) != 0) {
      const int id = id_of(ins.args[0]);
      if (id >= 0) defs[i].push_back(id);
    }
    for (size_t k = 1; k < ins.args.size(); ++k) {
      const int id = id_of(ins.args[k]);
      if (id >= 0) uses[i].push_back(id);
    }
    if (i + 1 < n) succ[i].push_back(static_cast<int>(i + 1));
  }

  const int R = static_cast<int>(weights.size());
  std::vector<std::vector<char>> liveIn(n, std::vector<char>(R, 0));
  std::vector<std::vector<char>> liveOut(n, std::vector<char>(R, 0));
  bool changed = true;
  for (int iter = 0; iter < 100000 && changed; ++iter) {
    changed = false;
    for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
      for (int s : succ[i])
        for (int r = 0; r < R; ++r)
          if (liveIn[s][r] && !liveOut[i][r]) {
            liveOut[i][r] = 1;
            changed = true;
          }
      for (int r = 0; r < R; ++r) {
        bool in_use = false;
        for (int u : uses[i])
          if (u == r) {
            in_use = true;
            break;
          }
        bool in_def = false;
        for (int d : defs[i])
          if (d == r) {
            in_def = true;
            break;
          }
        const bool v = in_use || (liveOut[i][r] && !in_def);
        if (v && !liveIn[i][r]) {
          liveIn[i][r] = 1;
          changed = true;
        }
      }
    }
  }
  int best = 0;
  for (size_t i = 0; i < n; ++i) {
    int w = 0;
    for (int r = 0; r < R; ++r)
      if (liveOut[i][r]) w += weights[r];
    if (w > best) best = w;
  }
  return best;
}

}  // namespace

// Находит (программу, индекс) по mangled-имени: сначала runtime-модули
// (ROUGE_CUDA_PTX + fatbin), потом driver-модули (cuModuleLoadData).
bool find_kernel_program(const std::string& name, const PtxProgram** prog, int* fnIndex) {
  for (const auto& mod : runtime_modules()) {
    const auto it = mod->functionIndex.find(name);
    if (it != mod->functionIndex.end()) {
      *prog = mod.get();
      *fnIndex = it->second;
      return true;
    }
  }
  for (const auto& kv : ctx().modules) {
    const auto it = kv.second->functionIndex.find(name);
    if (it != kv.second->functionIndex.end()) {
      *prog = kv.second.get();
      *fnIndex = it->second;
      return true;
    }
  }
  return false;
}

bool kernel_stats(const PtxProgram& prog, int fnIndex, KernelStats* out) {
  if (out == nullptr || fnIndex < 0 ||
      fnIndex >= static_cast<int>(prog.functions.size()))
    return false;
  const PtxFunction& fn = prog.functions[fnIndex];
  KernelStats st;
  st.numRegs = max_live_regs(fn);  // 0 при откате? нет: см. ниже
  if (st.numRegs <= 0) st.numRegs = declared_regs_bound(fn);
  if (st.numRegs <= 0) st.numRegs = 1;
  st.staticSmem = fn.sharedSize > 0 ? static_cast<size_t>(fn.sharedSize) : 0;
  st.localSize = fn.localSize > 0 ? static_cast<size_t>(fn.localSize) : 0;
  for (const auto& g : prog.globals)
    if (g.isConst)
      st.constSize += static_cast<size_t>(g.elemBytes) *
                      static_cast<size_t>(g.count > 0 ? g.count : 1);
  st.targetSm = prog.targetSm;
  *out = st;
  return true;
}

int occ_active_blocks(const KernelStats& st, int blockSize, size_t dynamicSmem) {
  if (blockSize <= 0 || blockSize > kMaxThreadsPerBlock) return 0;
  const int warpsPerCta = divUp(blockSize, kWarpSize);

  // Лимит варпов: слоты SM делятся на варпы блока.
  const int maxWarpsPerSm = kMaxThreadsPerSm / kWarpSize;
  const int limitWarps = maxWarpsPerSm / warpsPerCta;

  // Лимит регистров: аллокация идёт на варп с гранулярностью 256.
  // Проверка аппаратуры (assumed) округляет варпы до числа партиций.
  const int regsPerWarp = st.numRegs * kWarpSize;
  const int regsAllocatedPerWarp = roundUp(regsPerWarp, kRegAllocGranularity);
  const int regsAllocatedPerCta = regsAllocatedPerWarp * warpsPerCta;
  const int regsAssumedPerCta = regsAllocatedPerWarp *
                                roundUp(warpsPerCta, kSubPartitions);
  int limitRegs = 0;
  if (kRegsPerBlock >= regsAssumedPerCta && kRegsPerBlock >= regsAllocatedPerCta &&
      st.numRegs <= kMaxRegsPerThread && regsAllocatedPerWarp > 0) {
    const int regsPerSubPartition = kRegsPerSm / kSubPartitions;
    const int warpsPerSubPartition = regsPerSubPartition / regsAllocatedPerWarp;
    limitRegs = (warpsPerSubPartition * kSubPartitions) / warpsPerCta;
  }

  // Лимит блоков: архитектурный максимум SM.
  const int limitBlocks = kMaxBlocksPerSm;

  // Лимит shared: static + dynamic, округление до гранулярности; дальше —
  // default carveout, а если не влез — Turing-buckets (как в заголовке:
  // на Volta+ запрошенный объём меньше CTA не режет occupancy, а поднимает
  // carveout до bucket'а).
  const size_t totalSmem = st.staticSmem + dynamicSmem;
  const size_t allocatedSmem =
      static_cast<size_t>(roundUp(static_cast<int>(totalSmem), 256));
  int limitSmem = 0;
  if (allocatedSmem <= kSmemPerBlockLimit) {
    size_t smemPerSm = kSmemPerSmDefault;
    if (smemPerSm < allocatedSmem) {
      smemPerSm = allocatedSmem;
      if (!alignUpTuring(&smemPerSm)) return 0;
    }
    limitSmem = allocatedSmem == 0 ? INT_MAX
                                   : static_cast<int>(smemPerSm / allocatedSmem);
  }

  int m = limitWarps;
  if (limitRegs < m) m = limitRegs;
  if (limitBlocks < m) m = limitBlocks;
  if (limitSmem < m) m = limitSmem;
  return m;
}

void occ_potential_block_size(int* minGridSize, int* blockSize, const KernelStats& st,
                              size_t (*blockSizeToDynamicSMem)(int), size_t dynamicSmemFixed,
                              int numSms) {
  int bestBlock = 0, bestNumBlocks = 0, bestThreads = 0;
  // Перебор сверху вниз с шагом в варп; при равенстве потоков побеждает
  // ПЕРВЫЙ ( больший) размер — строгое ">" в сравнении ниже. Early-out на
  // полном заполнении SM (1024 потока).
  for (int tryAligned = 1024; tryAligned > 0; tryAligned -= kWarpSize) {
    const int trySize = tryAligned <= kMaxThreadsPerBlock ? tryAligned : kMaxThreadsPerBlock;
    size_t dyn = dynamicSmemFixed;
    if (blockSizeToDynamicSMem != nullptr) dyn = blockSizeToDynamicSMem(trySize);
    const int active = occ_active_blocks(st, trySize, dyn);
    const int threads = trySize * active;
    if (threads > bestThreads) {
      bestThreads = threads;
      bestBlock = trySize;
      bestNumBlocks = active;
    }
    if (bestThreads == kMaxThreadsPerSm) break;
  }
  *blockSize = bestBlock;
  *minGridSize = bestNumBlocks * numSms;
}

bool occ_available_dynamic_smem(size_t* out, const KernelStats& st, int numBlocks,
                                int blockSize) {
  if (out == nullptr || numBlocks <= 0) return false;
  // Сначала occupancy потенциального запуска (dynamic = 0): если столько
  // блоков не влезет даже без динамики — отказ, как у настоящего драйвера.
  if (occ_active_blocks(st, blockSize, 0) < numBlocks) return false;
  size_t smemPerSm;
  if (numBlocks == 1) {
    smemPerSm = kSmemPerBlockLimit;
  } else {
    // User preference по умолчанию — полный default carveout; align-up до
    // bucket'а нужен только когда preference меньше (здесь он всегда полный).
    smemPerSm = kSmemPerSmDefault;
  }
  size_t avail = smemPerSm / static_cast<size_t>(numBlocks);
  avail = (avail / kSmemAllocGranularity) * kSmemAllocGranularity;
  if (avail > kSmemPerBlockLimit) avail = kSmemPerBlockLimit;
  if (avail < st.staticSmem) return false;
  avail -= st.staticSmem;
  // sm_75 < sm_80: вычета reservedSharedMemPerBlock нет (только с Ampere).
  const size_t maxDyn =
      st.staticSmem >= kSmemPerBlockLimit ? 0 : kSmemPerBlockLimit - st.staticSmem;
  if (avail > maxDyn) avail = maxDyn;
  *out = avail;
  return true;
}

}  // namespace rouge
