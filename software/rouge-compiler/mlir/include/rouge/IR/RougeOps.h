//===- RougeOps.h - the operations of the ROUGE dialect -*- C++ -*-===//
//
// The generated declarations of the operations (RougeOps.h.inc), of the
// dialect itself (RougeOpsDialect.h.inc), of the dimension enum
// (RougeOpsEnums.h.inc) and of the enum attribute (RougeOpsAttributes.h.inc).
//
// The order matters: the ops refer to the attribute class, so the attribute
// has to be declared first.
//
//===----------------------------------------------------------------------===//

#ifndef ROUGE_OPS_H
#define ROUGE_OPS_H

// The includes the generated code relies on: the builtin types (IndexType),
// the op definition machinery and the side-effect interfaces (the memory
// effects of load/store).
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "rouge/IR/RougeOpsEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "rouge/IR/RougeOpsAttributes.h.inc"

#include "rouge/IR/RougeOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "rouge/IR/RougeOps.h.inc"

#endif // ROUGE_OPS_H
