// ROUGE-V: freestanding RISC-V harness for the vectorised MLIR module.
//
// The module (test/unit_stride_to_vector.mlir) allocates its memrefs, so the
// lowering of memref.alloc calls malloc — here a bump allocator over a
// static buffer (no libc, no OS startup, exactly like rv_exec_harness.c).
//
// The result of the module is a float (the sum of the copied array) printed
// through print_f32; the harness exits with its integer part, so a runner
// (qemu-user) sees the number as the exit code and the scalar and the
// vectorised runs can be compared.
//
// The exit syscall: a7 = 93 (SYS_exit), a0 = status.

extern void main(void);

// The module calls this with the sum.
static float g_result = 0.0f;
void print_f32(float f) { g_result = f; }

// A bump allocator: 64 KiB is plenty for the test module.
static unsigned char g_heap[64 * 1024] __attribute__((aligned(16)));
static unsigned long g_heap_used = 0;

void *malloc(unsigned long size) {
  const unsigned long aligned = (size + 15ul) & ~15ul;
  if (g_heap_used + aligned > sizeof(g_heap)) return (void *)0;
  void *p = &g_heap[g_heap_used];
  g_heap_used += aligned;
  return p;
}

void free(void *p) { (void)p; }  // a bump allocator never frees

void _start(void) {
  main();
  // Linux exit(status).
  register long a0 __asm__("a0") = (long)g_result;
  register long a7 __asm__("a7") = 93;
  __asm__ volatile("ecall" : : "r"(a0), "r"(a7) : "memory");
  for (;;) {
  }
}
