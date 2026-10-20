//===- RougeDialect.cpp - ROUGE-V dialect -*- C++ -*-===//

#include "rouge/IR/RougeDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "rouge/IR/RougeOpsEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "rouge/IR/RougeOpsAttributes.cpp.inc"

#include "rouge/IR/RougeOpsDialect.cpp.inc"

using namespace mlir;
using namespace rouge;

#define GET_OP_CLASSES
#include "rouge/IR/RougeOps.cpp.inc"

void RougeDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "rouge/IR/RougeOps.cpp.inc"
      >();
  // The dimension attribute has to be registered: without it the attribute
  // storage uniquer is not initialized for it, and creating the attribute
  // crashes (in a release build, where the debug assertion is compiled out).
  addAttributes<DimensionAttr>();
}

// The dialect declares the type hooks (the generated header does it when the
// dialect says it uses the default printer/parser), so they have to be
// defined. The dialect has no custom type syntax yet; the attribute hooks
// are already defined by the generated attribute code.
::mlir::Type RougeDialect::parseType(::mlir::DialectAsmParser &parser) const {
  return ::mlir::Dialect::parseType(parser);
}

void RougeDialect::printType(::mlir::Type type,
                             ::mlir::DialectAsmPrinter &os) const {
  ::mlir::Dialect::printType(type, os);
}
