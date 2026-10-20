# ROUGE-V: our own MLIR dialect (the "rouge" dialect).
#
# This is the foundation of the 2.0 plan: PTX -> rouge dialect -> (passes) ->
# LLVM IR. The dialect describes what is specific to a SIMT kernel — the
# thread/block identity, the sizes, the barrier and the memory access with an
# explicit address space — and leaves the element-wise arithmetic to the
# upstream `arith` dialect.
#
# The ops are generated with mlir-tblgen from include/rouge/IR/RougeOps.td.

set(MLIR_TABLEGEN_EXE mlir-tblgen)

# This file lives in mlir/tools/rouge-dialect/, so the paths to the dialect
# are two levels up.
set(ROUGE_DIALECT_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
set(ROUGE_DIALECT_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/../../include)
set(ROUGE_DIALECT_TD ${ROUGE_DIALECT_INC_DIR}/rouge/IR/RougeOps.td)

# --- the generated headers -------------------------------------------------
set(ROUGE_TABLEGEN_OUTPUTS
  ${ROUGE_DIALECT_GEN_DIR}/rouge/IR/RougeOps.h.inc
  ${ROUGE_DIALECT_GEN_DIR}/rouge/IR/RougeOps.cpp.inc
  ${ROUGE_DIALECT_GEN_DIR}/rouge/IR/RougeOpsDialect.h.inc
  ${ROUGE_DIALECT_GEN_DIR}/rouge/IR/RougeOpsDialect.cpp.inc
  ${ROUGE_DIALECT_GEN_DIR}/rouge/IR/RougeOpsEnums.h.inc
  ${ROUGE_DIALECT_GEN_DIR}/rouge/IR/RougeOpsEnums.cpp.inc
  ${ROUGE_DIALECT_GEN_DIR}/rouge/IR/RougeOpsAttributes.h.inc
  ${ROUGE_DIALECT_GEN_DIR}/rouge/IR/RougeOpsAttributes.cpp.inc)

add_custom_command(
  OUTPUT ${ROUGE_TABLEGEN_OUTPUTS}
  COMMAND ${CMAKE_COMMAND} -E make_directory
          ${ROUGE_DIALECT_GEN_DIR}/rouge/IR
  COMMAND ${MLIR_TABLEGEN_EXE} -gen-op-decls
          -I ${ROUGE_DIALECT_INC_DIR}
          -I ${MLIR_INCLUDE_DIRS}
          ${ROUGE_DIALECT_TD}
          -o ${ROUGE_DIALECT_GEN_DIR}/rouge/IR/RougeOps.h.inc
  COMMAND ${MLIR_TABLEGEN_EXE} -gen-op-defs
          -I ${ROUGE_DIALECT_INC_DIR}
          -I ${MLIR_INCLUDE_DIRS}
          ${ROUGE_DIALECT_TD}
          -o ${ROUGE_DIALECT_GEN_DIR}/rouge/IR/RougeOps.cpp.inc
  COMMAND ${MLIR_TABLEGEN_EXE} -gen-dialect-decls
          -dialect=rouge
          -I ${ROUGE_DIALECT_INC_DIR}
          -I ${MLIR_INCLUDE_DIRS}
          ${ROUGE_DIALECT_TD}
          -o ${ROUGE_DIALECT_GEN_DIR}/rouge/IR/RougeOpsDialect.h.inc
  COMMAND ${MLIR_TABLEGEN_EXE} -gen-dialect-defs
          -dialect=rouge
          -I ${ROUGE_DIALECT_INC_DIR}
          -I ${MLIR_INCLUDE_DIRS}
          ${ROUGE_DIALECT_TD}
          -o ${ROUGE_DIALECT_GEN_DIR}/rouge/IR/RougeOpsDialect.cpp.inc
  COMMAND ${MLIR_TABLEGEN_EXE} -gen-enum-decls
          -I ${ROUGE_DIALECT_INC_DIR}
          -I ${MLIR_INCLUDE_DIRS}
          ${ROUGE_DIALECT_TD}
          -o ${ROUGE_DIALECT_GEN_DIR}/rouge/IR/RougeOpsEnums.h.inc
  COMMAND ${MLIR_TABLEGEN_EXE} -gen-enum-defs
          -I ${ROUGE_DIALECT_INC_DIR}
          -I ${MLIR_INCLUDE_DIRS}
          ${ROUGE_DIALECT_TD}
          -o ${ROUGE_DIALECT_GEN_DIR}/rouge/IR/RougeOpsEnums.cpp.inc
  COMMAND ${MLIR_TABLEGEN_EXE} -gen-attrdef-decls
          -I ${ROUGE_DIALECT_INC_DIR}
          -I ${MLIR_INCLUDE_DIRS}
          ${ROUGE_DIALECT_TD}
          -o ${ROUGE_DIALECT_GEN_DIR}/rouge/IR/RougeOpsAttributes.h.inc
  COMMAND ${MLIR_TABLEGEN_EXE} -gen-attrdef-defs
          -I ${ROUGE_DIALECT_INC_DIR}
          -I ${MLIR_INCLUDE_DIRS}
          ${ROUGE_DIALECT_TD}
          -o ${ROUGE_DIALECT_GEN_DIR}/rouge/IR/RougeOpsAttributes.cpp.inc
  DEPENDS ${ROUGE_DIALECT_TD}
  COMMENT "[rouge-mlir] mlir-tblgen: rouge dialect")

# A plain target over the generated files: the MLIR CMake helper
# (add_public_tablegen_target) lives in AddMLIR.cmake, which the package
# config we use does not pull in.
add_custom_target(rouge-dialect-headers DEPENDS ${ROUGE_TABLEGEN_OUTPUTS})
