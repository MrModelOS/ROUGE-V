// ptx2ir — ROUGE-V AOT compiler, stage 1 CLI.
// Usage: ptx2ir <kernel.ptx> [output.ll]
// Parses PTX assembly and emits textual LLVM IR for the first (.entry) kernel.
// With no output path the IR goes to stdout.

#include <fstream>
#include <iostream>
#include <sstream>

#include "ptx.h"
#include "rougecomp/ptx_to_llvm.h"

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    std::cerr << "usage: " << argv[0] << " <kernel.ptx> [output.ll]\n";
    return 2;
  }
  const std::string inPath = argv[1];
  const std::string outPath = (argc > 2) ? argv[2] : "";

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
  std::string ir = rougecomp::ptx_to_llvm_ir(*prog, 0, &err);
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