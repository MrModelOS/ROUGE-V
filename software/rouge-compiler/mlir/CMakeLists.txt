# ROUGE-V MLIR contour — stage 2 of the AOT pipeline (GPU -> Vector -> RVV).
#
# Enabled from the parent with -DROUGE_ENABLE_MLIR=ON. Requires an MLIR package
# (MLIR_DIR or an in-tree LLVM build): the host toolchain used for the
# PTX -> LLVM IR path ships clang without MLIR, so the contour is OFF by
# default and ctest simply has no MLIR tests to run.

find_package(MLIR CONFIG QUIET)

if(NOT MLIR_FOUND)
  message(STATUS "rouge MLIR contour: MLIR package not found — skipping")
  return()
endif()

message(STATUS "rouge MLIR contour: building against MLIR ${MLIR_VERSION}")

# ---- pass library ------------------------------------------------------
add_library(rouge-mlir-passes STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/Transforms/SimtAccessReport.cpp)
target_include_directories(rouge-mlir-passes PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS})
target_link_libraries(rouge-mlir-passes PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport)
target_compile_options(rouge-mlir-passes PRIVATE ${LLVM_COMPILE_OPTIONS})
if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(rouge-mlir-passes PUBLIC -fno-rtti)
endif()

# ---- rouge-opt driver --------------------------------------------------
add_executable(rouge-opt ${CMAKE_CURRENT_SOURCE_DIR}/tools/rouge-opt/rouge-opt.cpp)
target_link_libraries(rouge-opt PRIVATE
  rouge-mlir-passes
  MLIROptLib
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRFuncDialect
  MLIRArithDialect
  MLIRMemRefDialect)

# ---- tests ------------------------------------------------------------
add_test(NAME mlir_simt_access_report
  COMMAND rouge-opt --rouge-simt-access-report
          ${CMAKE_CURRENT_SOURCE_DIR}/test/simt_access_report.mlir)
