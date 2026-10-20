// ROUGE-V differential fuzzer: the REFERENCE side (our PTX interpreter).
//
// Takes the generated kernel (PTX text), the input data, and writes the
// interpreter's output. The GPU runner (run_gpu.c) takes the SAME kernel —
// but translated by OUR ptx2ir — and writes the card's output. The driver
// compares the two byte for byte.
//
// usage: run_interp <kernel.ptx> <in.bin> <out.bin> <n>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ptx.h"

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: %s <kernel.ptx> <in.bin> <out.bin> <n>\n",
                 argv[0]);
    return 2;
  }
  const char* ptxPath = argv[1];
  const char* inPath = argv[2];
  const char* outPath = argv[3];
  const int n = atoi(argv[4]);

  FILE* f = fopen(ptxPath, "rb");
  if (!f) {
    std::fprintf(stderr, "run_interp: cannot open %s\n", ptxPath);
    return 2;
  }
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  std::string text((size_t)sz, '\0');
  if (fread(&text[0], 1, text.size(), f) != text.size()) {
    std::fprintf(stderr, "run_interp: short read\n");
    return 2;
  }
  fclose(f);

  std::string err;
  auto prog = rouge::parse_ptx(text, &err);
  if (!prog) {
    std::fprintf(stderr, "run_interp: parse failed: %s\n", err.c_str());
    return 2;
  }
  int fnIndex = -1;
  for (size_t i = 0; i < prog->functions.size(); ++i)
    if (prog->functions[i].name == "fuzz_kernel") fnIndex = (int)i;
  if (fnIndex < 0) {
    std::fprintf(stderr, "run_interp: no fuzz_kernel entry\n");
    return 2;
  }
  const auto& fn = prog->functions[fnIndex];

  std::vector<int> input((size_t)n);
  f = fopen(inPath, "rb");
  if (!f || fread(input.data(), sizeof(int), input.size(), f) != input.size()) {
    std::fprintf(stderr, "run_interp: cannot read %s\n", inPath);
    return 2;
  }
  fclose(f);

  // Выходной буфер: n*K элементов (K = 4 в генераторе).
  const int kOut = 4;
  std::vector<int> output((size_t)n * kOut, 0);

  // Параметры ядра: (out, in, n, seed). Блоб — по раскладке парсера.
  std::vector<uint8_t> blob((size_t)fn.paramsBlobSize, 0);
  const unsigned seed = 12345u;
  for (const auto& p : fn.params) {
    if (p.index == 0) {
      const unsigned long long v = (unsigned long long)output.data();
      std::memcpy(blob.data() + p.offset, &v, 8);
    } else if (p.index == 1) {
      const unsigned long long v = (unsigned long long)input.data();
      std::memcpy(blob.data() + p.offset, &v, 8);
    } else if (p.index == 2) {
      const unsigned v = (unsigned)n;
      std::memcpy(blob.data() + p.offset, &v, 4);
    } else if (p.index == 3) {
      const unsigned v = seed;
      std::memcpy(blob.data() + p.offset, &v, 4);
    }
  }

  // Сетка: один блок из n потоков (как в GPU-запуске).
  if (!rouge::execute_kernel(*prog, fnIndex, blob, 1, 1, 1, (unsigned)n, 1, 1,
                             &err, nullptr)) {
    std::fprintf(stderr, "run_interp: exec failed: %s\n", err.c_str());
    return 2;
  }

  f = fopen(outPath, "wb");
  if (!f) {
    std::fprintf(stderr, "run_interp: cannot write %s\n", outPath);
    return 2;
  }
  std::fwrite(output.data(), sizeof(int), output.size(), f);
  fclose(f);
  return 0;
}
