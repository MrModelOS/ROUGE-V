// rouge-opt — MLIR driver for the ROUGE-V contour passes.
//
// Mirrors mlir-opt but registers the ROUGE passes (rouge-simt-access-report).
// Built only when the project is configured with -DROUGE_ENABLE_MLIR=ON and an
// MLIR package is found (see mlir/CMakeLists.txt).

#include "rouge/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#if __has_include("mlir/InitAllDialects.h")
#include "mlir/InitAllDialects.h"
#endif
#if __has_include("mlir/InitAllExtensions.h")
#include "mlir/InitAllExtensions.h"
#endif

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
#if __has_include("mlir/InitAllDialects.h")
  mlir::registerAllDialects(registry);
#else
  registry.insert<mlir::func::FuncDialect, mlir::arith::ArithDialect,
                  mlir::memref::MemRefDialect>();
#endif
#if __has_include("mlir/InitAllExtensions.h")
  mlir::registerAllExtensions(registry);
#endif

  rouge::registerRougeTransformsPasses();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "ROUGE-V MLIR contour optimizer\n", registry));
}
