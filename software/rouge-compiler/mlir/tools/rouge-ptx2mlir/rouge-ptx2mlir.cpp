//===- rouge-ptx2mlir.cpp - the PTX front-end of the rouge dialect -*- C++ -*-===//
//
// The tool: PTX in, a module of the `rouge` dialect out.
//
//   rouge-ptx2mlir kernel.ptx -o kernel.mlir
//
// The options:
//
//   --shape N=64xf32   the shape of the N-th kernel parameter (the PTX does
//                      not say how big the array is; only the caller knows).
//                      Without the hint the shape is dynamic, and the
//                      vectoriser (which needs a static shape) refuses.
//   --grid N --block M the launch configuration. With a block size, a host
//                      function with a rouge.launch is emitted as well: the
//                      launch configuration belongs to the caller.
//
// The exit code is 0 when the kernel was translated and 1 when it was
// REFUSED — the refusal is the point: a kernel that is translated into
// something else computes the wrong thing silently.
//
//===----------------------------------------------------------------------===//

#include "rouge/Frontend/PtxToRouge.h"

#include "rouge/IR/RougeDialect.h"

#include "ptx.h"  // the shared PTX parser

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include "llvm/Support/raw_ostream.h"
#include <string>

using namespace mlir;

int main(int argc, char **argv) {
  std::string input;
  std::string output;
  rouge::PtxShapeHints hints;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-o" && i + 1 < argc) {
      output = argv[++i];
    } else if (arg == "--grid" && i + 1 < argc) {
      hints.grid = atoi(argv[++i]);
    } else if (arg == "--block" && i + 1 < argc) {
      hints.block = atoi(argv[++i]);
    } else if (arg == "--shape" && i + 1 < argc) {
      // "0=64xf32"
      std::string spec = argv[++i];
      size_t eq = spec.find('=');
      if (eq == std::string::npos) {
        std::fprintf(stderr, "rouge-ptx2mlir: bad --shape '%s'\n", spec.c_str());
        return 2;
      }
      int index = atoi(spec.substr(0, eq).c_str());
      std::string shape = spec.substr(eq + 1);
      hints.shapes[index] = shape;
      size_t x = shape.find('x');
      if (x != std::string::npos) hints.elemTypes[index] = shape.substr(x + 1);
    } else if (arg.rfind("--", 0) == 0) {
      std::fprintf(stderr, "rouge-ptx2mlir: unknown option '%s'\n", arg.c_str());
      return 2;
    } else {
      input = arg;
    }
  }
  if (input.empty()) {
    std::fprintf(stderr, "usage: rouge-ptx2mlir kernel.ptx [-o kernel.mlir]\n");
    return 2;
  }

  std::ifstream file(input);
  if (!file) {
    std::fprintf(stderr, "rouge-ptx2mlir: cannot open '%s'\n", input.c_str());
    return 2;
  }
  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string text = buffer.str();

  std::string error;
  std::unique_ptr<rouge::PtxProgram> program = rouge::parse_ptx(text, &error);
  if (!program) {
    std::fprintf(stderr, "rouge-ptx2mlir: cannot parse the PTX: %s\n",
                 error.c_str());
    return 1;
  }

  DialectRegistry registry;
  registry.insert<rouge::RougeDialect, arith::ArithDialect, func::FuncDialect,
                  math::MathDialect, memref::MemRefDialect, scf::SCFDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  OpBuilder builder(&context);
  auto module = ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(module.getBody());

  error.clear();
  if (failed(rouge::translatePtxToRouge(module, *program, hints, &error))) {
    std::fprintf(stderr, "rouge-ptx2mlir: REFUSED: %s\n", error.c_str());
    return 1;
  }
  if (getenv("ROUGE_PTX2MLIR_DEBUG")) module.dump();
  if (failed(verify(module))) {
    std::fprintf(stderr, "rouge-ptx2mlir: the translated module does not "
                         "verify\n");
    return 1;
  }

  if (output.empty()) {
    module.print(llvm::outs());
  } else {
    std::error_code ec;
    llvm::raw_fd_ostream out(output, ec);
    if (ec) {
      std::fprintf(stderr, "rouge-ptx2mlir: cannot write '%s'\n", output.c_str());
      return 2;
    }
    module.print(out);
    out.flush();
  }
  return 0;
}
