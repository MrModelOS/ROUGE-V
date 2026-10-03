// Волна 1: настоящая CUDA-программа, собранная nvcc, исполняется нашим шимом.
//
// Это единственный тест, где проверяется не ядро, а ПРОГРАММА целиком:
// компилятор NVIDIA генерирует статическую регистрацию (fatbin + callback +
// __cudaRegisterFunction) и сахар <<<>>> поверх __cudaPushCallConfiguration и
// __cudaLaunchKernel. Мы перехватываем ровно этот протокол и исполняем PTX на
// CPU, поэтому успех означает «чужая программа считает без NVIDIA».
//
// Требует nvcc для СБОРКИ .cu и .ptx. Без nvcc тест честно пропускается;
// исполнение GPU не требуется вовсе — это и есть смысл проверки.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <dlfcn.h>

static int fail(const char* what) {
  std::fprintf(stderr, "FAIL: %s\n", what);
  return 1;
}

// Strips the compat layer's own diagnostics ("[ROUGE-CUDA] ...") so the
// comparison is about the PROGRAM's output only, not about our logging.
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

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s <prog_ref> <prog_shimmed> <shim.so>\n",
                 argv[0]);
    return 2;
  }
  // prog_ref  : the same program linked against the real CUDA runtime
  // prog      : the same object linked against OUR shim, no CUDA library at all
  // The shimmed binary must find its kernels INSIDE ITSELF: no PTX file and no
  // environment variable are handed to it.
  const std::string progRef = argv[1];
  const std::string prog = argv[2];
  const std::string shim = argv[3];

  // Run the SAME binary twice: once against the real libcudart (if the host has
  // it, i.e. a genuine NVIDIA stack) and once against our libcuda imposter with
  // emulation forced. Comparing the two is the whole point; when there is no
  // real driver the second run stands alone and must simply pass.
  // Work next to the binaries themselves: both are produced in WORK_DIR.
  const std::string cwd = prog.substr(0, prog.find_last_of('/'));

  // Reference run: no LD_LIBRARY_PATH interference.
  const std::string refCmd =
      "cd " + cwd + " && ./" + progRef.substr(progRef.find_last_of('/') + 1) +
      " > ref.out 2>&1; echo rc=$?";
  const int refRc = std::system(refCmd.c_str());

  // Emulation run: our library as libcuda.so.1, PTX supplied out of band.
  const std::string shimDir = cwd + "/rouge-imposter";
  if (std::system(("mkdir -p " + shimDir).c_str()) != 0) return fail("mkdir failed");
  const std::string copyCmd = "cp -f " + shim + " " + shimDir + "/libcuda.so.1";
  if (std::system(copyCmd.c_str()) != 0) return fail("cannot install the shim as libcuda.so.1");

  const std::string emuCmd = "cd " + cwd + " && LD_LIBRARY_PATH=" + shimDir +
                             " ROUGE_CUDA_FORCE_EMU=1 ./" +
                             prog.substr(prog.find_last_of('/') + 1) +
                             " > emu.out 2>&1; echo rc=$?";
  const int emuRc = std::system(emuCmd.c_str());

  std::string refOut, emuOut;
  if (!read_file(cwd + "/ref.out", &refOut)) return fail("no reference output");
  if (!read_file(cwd + "/emu.out", &emuOut)) return fail("no emulation output");

  std::printf("--- reference: same program, real CUDA runtime ---\n%s", refOut.c_str());
  std::printf("--- emulation (ROUGE shim, %s) ---\n%s", shim.c_str(), emuOut.c_str());

  // The emulated run must succeed.
  if (emuRc != 0) {
    std::fprintf(stderr, "FAIL: emulated run exited %d\n", emuRc);
    return 1;
  }
  if (emuOut.find("OK") == std::string::npos) {
    std::fprintf(stderr, "FAIL: program did not report OK under emulation\n");
    return 1;
  }
  // The real-CUDA binary and the shimmed one must agree exactly.
  if (refRc != 0) {
    std::fprintf(stderr, "FAIL: reference run on the real stack exited %d\n", refRc);
    return 1;
  }
  if (program_output_only(refOut) != program_output_only(emuOut)) {
    std::fprintf(stderr, "FAIL: emulation output differs from the real stack\n");
    return 1;
  }
  std::printf("PASS: foreign CUDA program runs without NVIDIA (program, not just a kernel)\n");
  return 0;
}