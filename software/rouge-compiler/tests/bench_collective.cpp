// Бенчмарк варп-коллективов: AOT против интерпретатора, честный замер.
//
// Один блок 32 потока как постоянная команда OS-тредов (два барьерных раунда
// на итерацию вместо spawn/join — меряется счёт, а не создание тредов).
// Интерпретатор — один тред, round-robin, парсинг один раз снаружи.
// Печатает микросекунды на запуск и ускорение; всегда exit 0 (это замер,
// а не проверка — проверяют test_aot_*).
//
// Собирается из уже построенных .o канонических ядер (mma_pipe.o,
// ldmatrix_x4.o) плюс интерпретатор из исходников.

#include <pthread.h>
#include <rouge_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "ptx.h"

extern "C" void mma_pipe(uint64_t out, const void* launch);
extern "C" int32_t __rouge_mma_pipe_query(RougeKernelInfo* info);
extern "C" void ldmatrix_x4(uint64_t out, const void* launch);
extern "C" int32_t __rouge_ldmatrix_x4_query(RougeKernelInfo* info);

namespace {

struct Launch {
  int32_t tidx, tiy, tiz;
  int32_t ctaidx, ctaidy, ctaidz;
  int32_t ntidx, ntidy, ntidz;
  int32_t nctaidx, nctaidy, nctaidz;
  uint64_t smem;
  void* barrier;
  void* exchange;
};

constexpr int32_t kThreads = 32;
constexpr int kIters = 200;
constexpr int kWarmup = 20;

typedef void (*KernelFn)(uint64_t, const void*);

struct Team {
  pthread_barrier_t start;
  pthread_barrier_t finish;
  KernelFn kernel = nullptr;
  float* out = nullptr;
  std::vector<Launch> launches;
  struct BlockState {
    pthread_barrier_t barrier;
    void* scratch = nullptr;
    void* exchange = nullptr;
  };
  BlockState block;
};

void worker(Team* tm, int t) {
  Launch& L = tm->launches[(size_t)t];
  L = Launch{};
  L.tidx = t;
  L.ntidx = kThreads;
  L.nctaidx = 1;
  L.smem = reinterpret_cast<uint64_t>(tm->block.scratch);
  L.barrier = &tm->block.barrier;
  L.exchange = tm->block.exchange;
  for (int i = 0; i < kIters + kWarmup; ++i) {
    pthread_barrier_wait(&tm->start);
    tm->kernel(reinterpret_cast<uint64_t>(tm->out), &L);
    pthread_barrier_wait(&tm->finish);
  }
}

// Возвращает микросекунды на запуск (без warmup).
double bench_aot(Team& tm, KernelFn kernel, float* out) {
  tm.kernel = kernel;
  tm.out = out;
  pthread_barrier_init(&tm.start, nullptr, kThreads + 1);
  pthread_barrier_init(&tm.finish, nullptr, kThreads + 1);
  tm.launches.assign(kThreads, Launch{});
  std::vector<std::thread> ths;
  for (int t = 0; t < kThreads; ++t)
    ths.emplace_back(worker, &tm, t);
  // Warmup внутри worker-цикла (первые kWarmup итераций не меряем).
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < kIters + kWarmup; ++i) {
    pthread_barrier_wait(&tm.start);
    pthread_barrier_wait(&tm.finish);
    if (i == kWarmup - 1) t0 = std::chrono::steady_clock::now();
  }
  auto t1 = std::chrono::steady_clock::now();
  for (auto& th : ths) th.join();
  pthread_barrier_destroy(&tm.start);
  pthread_barrier_destroy(&tm.finish);
  const double us =
      std::chrono::duration<double, std::micro>(t1 - t0).count() / kIters;
  return us;
}

double bench_interp(const rouge::PtxProgram& prog, int fnIndex, float* out,
                    size_t floats) {
  const uint64_t dev = reinterpret_cast<uint64_t>(out);
  std::vector<uint8_t> blob(sizeof(uint64_t), 0);
  std::memcpy(blob.data(), &dev, sizeof(uint64_t));
  std::string err;
  for (int i = 0; i < kWarmup; ++i)
    rouge::execute_kernel(prog, fnIndex, blob, 1, 1, 1, kThreads, 1, 1, &err,
                          nullptr);
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < kIters; ++i)
    rouge::execute_kernel(prog, fnIndex, blob, 1, 1, 1, kThreads, 1, 1, &err,
                          nullptr);
  auto t1 = std::chrono::steady_clock::now();
  (void)floats;
  return std::chrono::duration<double, std::micro>(t1 - t0).count() / kIters;
}

struct LoadedProg {
  std::string text;
  std::unique_ptr<rouge::PtxProgram> prog;
};

static std::unique_ptr<LoadedProg> load_prog(const std::string& path) {
  auto lp = std::make_unique<LoadedProg>();
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return nullptr;
  std::fseek(f, 0, SEEK_END);
  long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  lp->text.assign((size_t)n, '\0');
  if (std::fread(&lp->text[0], 1, lp->text.size(), f) != lp->text.size()) {
    std::fclose(f);
    return nullptr;
  }
  std::fclose(f);
  std::string err;
  lp->prog = rouge::parse_ptx(lp->text, &err);
  if (!lp->prog) return nullptr;
  return lp;
}

int fn_index(const rouge::PtxProgram& prog, const char* name) {
  for (size_t i = 0; i < prog.functions.size(); ++i)
    if (prog.functions[i].name == name) return (int)i;
  return -1;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : "kernels";
  std::string err;
  auto pipeProg = load_prog(dir + "/mma_pipe.ptx");
  auto ldmProg = load_prog(dir + "/ldmatrix_tile.ptx");
  if (!pipeProg || !ldmProg) {
    std::printf("BENCH SKIP: no kernel files\n");
    return 0;
  }

  std::printf("%-18s %12s %12s %8s\n", "kernel", "aot/us", "interp/us",
              "speedup");
  const struct {
    const char* name;
    KernelFn kernel;
    rouge_kernel_query_fn query;
    const rouge::PtxProgram* prog;  // points into LoadedProg owners below
    const char* entry;
    size_t floats;
  } cases[] = {
      {"mma_pipe", &mma_pipe, &__rouge_mma_pipe_query, pipeProg->prog.get(),
       "mma_pipe", 128},
      {"ldmatrix_x4", &ldmatrix_x4, &__rouge_ldmatrix_x4_query,
       ldmProg->prog.get(), "ldmatrix_x4", 256},
  };
  for (const auto& cs : cases) {
    RougeKernelInfo info{};
    if (!rouge_kernel_info_query(cs.query, &info)) {
      std::printf("%-18s query failed\n", cs.name);
      continue;
    }
    Team tm;
    pthread_barrier_init(&tm.block.barrier, nullptr, kThreads);
    tm.block.scratch = rouge_smem_alloc(&info);
    tm.block.exchange = rouge_exchange_alloc(kThreads);
    std::vector<float> aotOut(cs.floats, 0), interpOut(cs.floats, 0);
    const double aot = bench_aot(tm, cs.kernel, aotOut.data());
    const double interp =
        bench_interp(*cs.prog, fn_index(*cs.prog, cs.entry), interpOut.data(),
                     cs.floats);
    // Санити: первые слова должны совпасть (иначе замер бессмыслен).
    bool same = true;
    for (size_t i = 0; i < 8 && i < cs.floats; ++i)
      if (aotOut[i] != interpOut[i]) same = false;
    std::printf("%-18s %12.1f %12.1f %7.1fx%s\n", cs.name, aot, interp,
                interp / aot, same ? "" : "  MISMATCH!");
    if (tm.block.scratch) rouge_smem_free(tm.block.scratch);
    rouge_exchange_free(tm.block.exchange);
    pthread_barrier_destroy(&tm.block.barrier);
  }
  return 0;
}
