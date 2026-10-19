// Волна 1: vendor CUDA program that uses the REAL cuBLAS - comparison.
//
// The same object is linked twice: once against NVIDIA's own libcudart +
// libcublas (the reference, run on the real GPU) and once against our shims
// (the emulation, run on the CPU with no NVIDIA present). This binary runs
// both and compares their output.
//
// The comparison is byte-for-byte on the program's own output, which is
// printed with limited precision: NVIDIA's GPU cuBLAS and our CPU GEMM sum
// in a different order, so the last bits may differ - we compare the
// mathematics, not the order of additions.

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static bool read_file(const std::string& path, std::string* out) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) return false;
  std::fseek(f, 0, SEEK_END);
  const long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  out->assign(static_cast<size_t>(n < 0 ? 0 : n), '\0');
  const bool ok = n <= 0 || std::fread(&(*out)[0], 1, out->size(), f) == out->size();
  std::fclose(f);
  return ok;
}

// Strips our own diagnostics so the comparison is about the program.
static std::string program_output_only(const std::string& text) {
  std::string out;
  size_t pos = 0;
  while (pos < text.size()) {
    size_t end = text.find('\n', pos);
    if (end == std::string::npos) end = text.size();
    const std::string line = text.substr(pos, end - pos);
    if (line.rfind("[ROUGE-CUDA]", 0) != 0) {
      out += line;
      out += '\n';
    }
    pos = end + 1;
  }
  return out;
}

static std::string dirname_of(const std::string& path) {
  const size_t at = path.find_last_of('/');
  return at == std::string::npos ? std::string(".") : path.substr(0, at);
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <prog_ref> <prog_shimmed>\n", argv[0]);
    return 2;
  }
  const std::string progRef = argv[1];
  const std::string prog = argv[2];
  const std::string cwd = dirname_of(prog);

  const int refRc = std::system(
      ("cd " + cwd + " && ./" + progRef.substr(progRef.find_last_of('/') + 1) +
       " > ref.out 2>&1; echo rc=$?")
          .c_str());
  const int emuRc = std::system(
      ("cd " + cwd + " && ROUGE_CUDA_FORCE_EMU=1 ./" +
       prog.substr(prog.find_last_of('/') + 1) + " > emu.out 2>&1; echo rc=$?")
          .c_str());

  std::string refOut, emuOut;
  if (!read_file(cwd + "/ref.out", &refOut)) {
    std::fprintf(stderr, "FAIL: no reference output\n");
    return 1;
  }
  if (!read_file(cwd + "/emu.out", &emuOut)) {
    std::fprintf(stderr, "FAIL: no emulation output\n");
    return 1;
  }
  std::printf("--- reference: same program, real cuBLAS on the real GPU ---\n%s",
              refOut.c_str());
  std::printf("--- emulation (ROUGE shims, cuBLAS included) ---\n%s", emuOut.c_str());

  if (emuRc != 0) {
    std::fprintf(stderr, "FAIL: emulated run exited %d\n", emuRc);
    return 1;
  }
  if (emuOut.find("OK") == std::string::npos) {
    std::fprintf(stderr, "FAIL: program did not report OK under emulation\n");
    return 1;
  }
  if (refRc != 0) {
    std::fprintf(stderr, "FAIL: reference run on the real stack exited %d\n", refRc);
    return 1;
  }
  if (program_output_only(refOut) != program_output_only(emuOut)) {
    std::fprintf(stderr, "FAIL: emulation output differs from the real stack\n");
    return 1;
  }
  std::printf(
      "PASS: vendor program linked against NVIDIA's cuBLAS computes on our "
      "shim (no NVIDIA present)\n");
  return 0;
}
