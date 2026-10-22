// ROUGE-V differential fuzzer: the HOST half (our AOT lowering).
//
// The AOT'd kernel is linked into this binary (one executable per generated
// kernel, built by the driver). The kernel is called once per thread — in
// REAL threads, not one after another: a kernel with a barrier (bar.sync)
// and with atomics is only correct when the threads really run at the same
// time, and a sequential loop would let one thread's store overtake another
// thread's atomic. The output is written for the driver to compare with the
// interpreter's.
//
// usage: run_host <in.bin> <out.bin> <n>

#include <pthread.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// The AOT-compiled kernel: the kernel's own parameters plus the launch
// descriptor the runtime passes.
extern "C" void fuzz_kernel(unsigned long long out, unsigned long long in,
                            unsigned int n, unsigned int seed, const void* launch);

namespace {
// The launch descriptor the AOT'd kernel reads (rouge_runtime.h):
//   0..44: 12 x i32 (tid/ctaid/ntid/nctaid xyz), 48: scratchpad,
//   56: barrier handle, 64: the warp exchange region.
struct Launch {
  int32_t spec[12] = {0};
  uint64_t smem = 0;
  void* barrier = nullptr;
  uint64_t exchange = 0;
};

// The state every thread needs.
int g_n = 0;
int g_outPerThread = 4;
int* g_output = nullptr;
const int* g_input = nullptr;
uint32_t g_seed = 12345u;

// The block barrier: bar.sync -> __rouge_syncthreads -> this.
pthread_barrier_t g_barrier;

// One thread of the block.
void* threadMain(void* arg) {
  const intptr_t tid = (intptr_t)arg;
  Launch L{};
  L.spec[0] = (int32_t)tid;  // tid.x
  L.spec[3] = 0;             // ctaid.x
  L.spec[6] = g_n;           // ntid.x
  L.spec[9] = 1;             // nctaid.x
  L.barrier = &g_barrier;
  fuzz_kernel((unsigned long long)g_output,
              (unsigned long long)g_input, (unsigned int)g_n, g_seed, &L);
  return nullptr;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s <in.bin> <out.bin> <n>\n", argv[0]);
    return 2;
  }
  const char* inPath = argv[1];
  const char* outPath = argv[2];
  const int n = atoi(argv[3]);
  const int kOut = 4;

  std::vector<int> input((size_t)n);
  FILE* f = std::fopen(inPath, "rb");
  if (!f || std::fread(input.data(), sizeof(int), input.size(), f) != input.size()) {
    std::fprintf(stderr, "run_host: cannot read %s\n", inPath);
    return 2;
  }
  std::fclose(f);

  std::vector<int> output((size_t)n * kOut, 0);
  g_n = n;
  g_output = output.data();
  g_input = input.data();

  // The block barrier: one arrival per thread.
  if (pthread_barrier_init(&g_barrier, nullptr, (unsigned)n) != 0) {
    std::fprintf(stderr, "run_host: cannot create the barrier\n");
    return 2;
  }
  std::vector<pthread_t> threads((size_t)n);
  for (int i = 0; i < n; ++i) {
    if (pthread_create(&threads[(size_t)i], nullptr, threadMain,
                       (void*)(intptr_t)i) != 0) {
      std::fprintf(stderr, "run_host: cannot create the thread %d\n", i);
      return 2;
    }
  }
  for (int i = 0; i < n; ++i) pthread_join(threads[(size_t)i], nullptr);
  pthread_barrier_destroy(&g_barrier);

  f = std::fopen(outPath, "wb");
  if (!f) {
    std::fprintf(stderr, "run_host: cannot write %s\n", outPath);
    return 2;
  }
  std::fwrite(output.data(), sizeof(int), output.size(), f);
  std::fclose(f);
  return 0;
}
