# Embeds a PTX source file verbatim into a generated C++ header.
# PTX files in this tree contain no ", \ or $ characters, so a raw string
# literal (R"()") round-trips byte-for-byte. Used so the interpreter test and
# the rouge-compiler AOT test consume one and the same kernel file.
#   -DPTX_FILE=<path to .ptx>
#   -DOUT_FILE=<path to generated .h>
#   -DVAR_NAME=<C++ variable name> (default: kBlockReducePtx)
if(NOT DEFINED VAR_NAME OR VAR_NAME STREQUAL "")
  set(VAR_NAME kBlockReducePtx)
endif()
file(READ "${PTX_FILE}" CONTENT)
file(WRITE "${OUT_FILE}" "// Generated from ${PTX_FILE} — do not edit.\n"
  "// Byte-identical PTX shared with the rouge-compiler AOT test\n"
  "// (one kernel, two execution paths).\n"
  "static const char* ${VAR_NAME} = R\"(${CONTENT})\";\n")
