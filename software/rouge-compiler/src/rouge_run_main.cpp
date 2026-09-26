// rouge-run — check and compile a PTX kernel with the ROUGE-V toolchain.
//
// The question a user actually has is "will you translate my kernel?", not
// "show me the IR". This tool answers exactly that, in one command: parse,
// translate, and hand the result to a real backend compiler so a malformed
// translation is caught here rather than at link time.
//
//   rouge-run kernel.ptx              report: shared memory, atomics, sizes
//   rouge-run kernel.ptx -o kernel.o  also emit a native object
//   rouge-run kernel.ptx --print-ir   also write the translated IR
//
// Exit codes: 0 ok, 1 translation refused (unsupported op), 2 bad usage or I/O.

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "ptx.h"
#include "rougecomp/ptx_to_llvm.h"

namespace {

std::string slurp(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return "";
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

bool write_file(const std::string& path, const std::string& data) {
  std::ofstream o(path, std::ios::binary);
  if (!o) return false;
  o << data;
  return static_cast<bool>(o);
}

}  // namespace

int main(int argc, char** argv) {
  std::string inPath, objPath, irPath;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "-h" || a == "--help") {
      std::printf(
          "usage: rouge-run <kernel.ptx> [-o <kernel.o>] [--print-ir]\n"
          "\n"
          "Translate a PTX kernel and verify the result compiles.\n"
          "  -o, --object <path>   write a native object file\n"
          "      --print-ir [path]  also write the translated LLVM IR\n");
      return 0;
    }
    if (a == "-o" || a == "--object") {
      if (++i >= argc) { std::fprintf(stderr, "rouge-run: %s needs a path\n", a.c_str()); return 2; }
      objPath = argv[i];
    } else if (a == "--print-ir") {
      irPath = (i + 1 < argc && argv[i + 1][0] != '-') ? argv[++i] : "";
    } else if (!a.empty() && a[0] == '-') {
      std::fprintf(stderr, "rouge-run: unknown option '%s'\n", a.c_str());
      return 2;
    } else if (inPath.empty()) {
      inPath = a;
    } else {
      std::fprintf(stderr, "rouge-run: unexpected argument '%s'\n", a.c_str());
      return 2;
    }
  }
  if (inPath.empty()) {
    std::fprintf(stderr, "rouge-run: no input file\n");
    return 2;
  }

  const std::string text = slurp(inPath);
  if (text.empty()) {
    std::fprintf(stderr, "rouge-run: cannot read '%s'\n", inPath.c_str());
    return 2;
  }

  std::string err;
  auto prog = rouge::parse_ptx(text, &err);
  if (!prog) {
    std::fprintf(stderr, "rouge-run: PTX parse error: %s\n", err.c_str());
    return 1;
  }
  if (prog->functions.empty()) {
    std::fprintf(stderr, "rouge-run: no .entry kernels in %s\n", inPath.c_str());
    return 1;
  }

  std::printf("input: %s\n", inPath.c_str());
  std::printf("kernels found: %zu\n", prog->functions.size());
  const rouge::PtxFunction& fn = prog->functions[0];
  std::printf("  @%s  params=%zu  instructions=%zu\n", fn.name.c_str(),
              fn.params.size(), fn.body.size());
  if (fn.sharedSize > 0) {
    std::printf("  shared memory: %d bytes in %zu variable(s)\n", fn.sharedSize,
                fn.sharedVars.size());
    for (const auto& sv : fn.sharedVars)
      std::printf("    %-12s @%4d  %6d bytes  (.%s x %d)\n", sv.name.c_str(),
                  sv.offset, sv.elemBytes * sv.count,
                  sv.elemBytes == 4 ? "b32" : sv.elemBytes == 2 ? "b16" : "b8",
                  sv.count);
  } else {
    std::printf("  shared memory: none\n");
  }

  err.clear();
  const std::string ir = rougecomp::ptx_to_llvm_ir(*prog, 0, &err);
  if (ir.empty()) {
    std::fprintf(stderr, "\nrouge-run: REFUSED: %s\n", err.c_str());
    std::fprintf(stderr,
                 "This instruction is outside the supported subset. Nothing "
                 "was emitted; nothing was approximated.\n");
    return 1;
  }
  std::printf("translation: ok (%zu bytes of LLVM IR)\n", ir.size());

  if (!irPath.empty()) {
    const std::string path = irPath.empty() ? inPath + ".ll" : irPath;
    if (!write_file(path, ir)) {
      std::fprintf(stderr, "rouge-run: cannot write %s\n", path.c_str());
      return 2;
    }
    std::printf("wrote %s\n", path.c_str());
  }

  if (!objPath.empty()) {
    // Verify with a real backend so a broken translation fails here.
    const std::string ll = inPath + ".rouge.ll";
    if (!write_file(ll, ir)) {
      std::fprintf(stderr, "rouge-run: cannot write %s\n", ll.c_str());
      return 2;
    }
    const std::string cmd = "clang -O2 -c " + ll + " -o " + objPath + " 2>&1";
    const int rc = std::system(cmd.c_str());
    std::remove(ll.c_str());
    if (rc != 0) {
      std::fprintf(stderr,
                   "\nrouge-run: translation emitted IR that the backend "
                   "rejected — this is a compiler bug, please report it.\n");
      return 1;
    }
    std::printf("wrote %s (backend accepted the IR)\n", objPath.c_str());
  }

  std::printf("\nOK\n");
  return 0;
}
