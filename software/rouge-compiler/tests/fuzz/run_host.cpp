// ROUGE-V differential fuzzer: the HOST half (our AOT lowering).
//
// The AOT'd kernel is linked into this binary (one executable per generated
// kernel, built by the driver). The kernel is called once per thread, the
// way the runtime does it, and its output is written for the driver to
// compare with the interpreter's.
//
// usage: run_host <in.bin> <out.bin> <n>

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
  for (int i = 0; i < n; ++i) {
    Launch L{};
    L.spec[0] = i;  // tid.x
    L.spec[3] = 0;  // ctaid.x
    L.spec[6] = n;  // ntid.x
    L.spec[9] = 1;  // nctaid.x
    fuzz_kernel((unsigned long long)output.data(),
                (unsigned long long)input.data(), (unsigned int)n, 12345u, &L);
  }

  f = std::fopen(outPath, "wb");
  if (!f) {
    std::fprintf(stderr, "run_host: cannot write %s\n", outPath);
    return 2;
  }
  std::fwrite(output.data(), sizeof(int), output.size(), f);
  std::fclose(f);
  return 0;
}
