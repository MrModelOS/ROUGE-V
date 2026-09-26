// ptx2ir — ROUGE-V AOT compiler, stage 1 CLI.
// Usage: ptx2ir [--target <llvm-triple>] <kernel.ptx> [output.ll]
// Parses PTX assembly and emits textual LLVM IR for the first (.entry) kernel.
// With no output path the IR goes to stdout.

#include <fstream>
#include <iostream>
#include <sstream>

#include "ptx.h"
#include "rougecomp/ptx_to_llvm.h"

int main(int argc, char** argv) {
  if (argc < 2 || argc > 5) {
    std::cerr << "usage: " << argv[0]
              << " [--target <llvm-triple>] <kernel.ptx> [output.ll]\n"
                 "\n"
                 "targets:\n"
                 "  (default)                x86_64-pc-linux-gnu     host\n"
                 "  riscv64-unknown-elf      RISC-V      (+ -march=rv64gcv)\n"
                 "  amdgcn-amd-amdhsa        AMD GPU     (+ -mcpu=gfx1100)\n"
                 "  nvptx64-nvidia-cuda      NVIDIA GPU  (+ -march=sm_XX)\n";
    return 2;
  }
  std::string target = "x86_64-pc-linux-gnu";
  int i = 1;
  // Accept both "--target X" and "--target=X".
  if (std::string(argv[i]) == "--target") {
    if (++i >= argc) { std::cerr << "ptx2ir: --target needs a value\n"; return 2; }
    target = argv[i++];
  } else if (std::string(argv[i]).rfind("--target=", 0) == 0) {
    target = std::string(argv[i]).substr(9);
    ++i;
  }
  if (i >= argc) { std::cerr << "ptx2ir: no input file\n"; return 2; }
  const std::string inPath = argv[i];
  const std::string outPath = (i + 1 < argc) ? argv[i + 1] : "";

  std::ifstream in(inPath, std::ios::binary);
  if (!in) {
    std::cerr << "ptx2ir: cannot open " << inPath << "\n";
    return 2;
  }
  std::stringstream buf;
  buf << in.rdbuf();

  std::string err;
  auto prog = rouge::parse_ptx(buf.str(), &err);
  if (!prog) {
    std::cerr << "ptx2ir: PTX parse error: " << err << "\n";
    return 1;
  }
  if (prog->functions.empty()) {
    std::cerr << "ptx2ir: no .entry kernels found in " << inPath << "\n";
    return 1;
  }

  err.clear();
  std::string ir = rougecomp::ptx_to_llvm_ir(*prog, 0, &err, target);
  if (ir.empty()) {
    std::cerr << "ptx2ir: translation error: " << err << "\n";
    return 1;
  }

  if (outPath.empty()) {
    std::cout << ir;
  } else {
    std::ofstream out(outPath, std::ios::binary);
    if (!out) {
      std::cerr << "ptx2ir: cannot write " << outPath << "\n";
      return 2;
    }
    out << ir;
  }
  return 0;
}