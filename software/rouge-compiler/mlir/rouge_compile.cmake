# Runs the backend driver (rouge-compile) on a PTX kernel and checks the
# assembly it produced. Structural test: the semantics of the same chain are
# covered by vector_rvv.cmake, which executes the result under qemu.
#
# The expected patterns are regexes: the point is that the vector unit was
# actually used, not that the assembler chose a particular register.
execute_process(
  COMMAND ${CMAKE_COMMAND} -E env
    ROUGE_OPT=${ROUGE_OPT}
    ROUGE_PTX2MLIR=${ROUGE_PTX2MLIR}
    MLIR_OPT=${MLIR_OPT}
    MLIR_TRANSLATE=${MLIR_TRANSLATE}
    LLC=${LLC}
    ${PYTHON} ${DRIVER} --target ${TARGET} ${ARGS} ${INPUT} -o ${OUTPUT}
  RESULT_VARIABLE rc
  OUTPUT_VARIABLE out
  ERROR_VARIABLE err)

if(NOT rc EQUAL 0)
  message(FATAL_ERROR "rouge-compile failed (${rc}):\n${out}\n${err}")
endif()

file(READ ${OUTPUT} asm)
foreach(pattern ${EXPECTED})
  if(NOT asm MATCHES "${pattern}")
    message(FATAL_ERROR
      "the ${TARGET} assembly does not contain '${pattern}':\n${asm}")
  endif()
endforeach()

message(STATUS "rouge-compile: the ${TARGET} assembly is what the test expects")
