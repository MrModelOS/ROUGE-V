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

#include "rouge/IR/RougeDialect.h"
#include "rouge/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::gpu::GPUDialect, mlir::memref::MemRefDialect,
                  mlir::arith::ArithDialect, mlir::vector::VectorDialect,
                  mlir::func::FuncDialect, mlir::scf::SCFDialect,
                  mlir::math::MathDialect, mlir::LLVM::LLVMDialect,
                  mlir::NVVM::NVVMDialect, rouge::RougeDialect>();

  // Имя регистрации — как в Passes.h (раньше драйвер звал несуществующее
  // registerRougeTransformsPasses; это всплыло только при первой реальной
  // сборке контура).
  rouge::registerRougeSimtTransformsPasses();
  rouge::registerRougeToGpuPass();
  rouge::registerRougeExtractKernelsPass();

  // The generic clean-up passes the contour uses: the vectoriser leaves an
  // empty scalar tail (a trip count that is a multiple of the vector width)
  // and the front-end can leave dead arithmetic. They are registered by
  // hand because this driver is not mlir-opt: it links only the dialects
  // and the passes the contour needs.
  mlir::registerCanonicalizerPass();
  mlir::registerCSEPass();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "ROUGE-V MLIR contour optimizer\n", registry));
}
