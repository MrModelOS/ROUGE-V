// rouge-opt — MLIR driver for the ROUGE-V contour passes.
//
// Mirrors mlir-opt but registers the ROUGE passes (rouge-simt-access-report).
// Built only when the project is configured with -DROUGE_ENABLE_MLIR=ON and an
// MLIR package is found (see mlir/CMakeLists.txt).
//
// Диалекты регистрируются точечно (gpu, memref, arith, vector, func), а не
// через InitAllDialects: тот тянет ВСЕ диалекты MLIR, а линковать их все
// ради одного анализа незачем. Набора выше хватает для форм, которые читает
// контур (см. test/simt_access_report.mlir).

#include "rouge/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::gpu::GPUDialect, mlir::memref::MemRefDialect,
                  mlir::arith::ArithDialect, mlir::vector::VectorDialect,
                  mlir::func::FuncDialect, mlir::scf::SCFDialect>();

  // Имя регистрации — как в Passes.h (раньше драйвер звал несуществующее
  // registerRougeTransformsPasses; это всплыло только при первой реальной
  // сборке контура).
  rouge::registerRougeSimtTransformsPasses();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "ROUGE-V MLIR contour optimizer\n", registry));
}
