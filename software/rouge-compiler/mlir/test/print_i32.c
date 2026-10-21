// ROUGE-V: the tiny runtime for the differential fuzzer of the MLIR contour.
//
// The module prints the values of the kernel's output, one per line, so that
// the driver can compare them with the PTX interpreter's output element by
// element (a sum would hide a permutation).
#include <stdio.h>

void print_i32(int v) { printf("%d\n", v); }
