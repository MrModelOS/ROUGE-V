// ROUGE-V: freestanding RISC-V harness — a translated kernel actually RUNS.
//
// The RISC-V target of ptx2ir was so far compile-only: the IR was accepted
// by the LLVM backend, but nothing executed it. This harness closes that gap
// for kernels that do not need cross-lane barriers (vadd has none):
//
//   * no libc, no threads, no startup code from the OS — the kernel body is
//     called lane by lane, with the launch descriptor updated per lane;
//   * the launch descriptor layout is the one ptx2ir emits (see
//     kSpecNames in ptx_to_llvm.cpp): offset 0 = tid.x, 12 = ctaid.x,
//     24 = ntid.x, 48 = scratchpad base, 56 = barrier handle,
//     64 = warp-exchange base;
//   * the result is checked and the process exits with 0 (ok) or 1 (mismatch)
//     through the Linux exit syscall, so a runner (qemu-user) sees the code.
//
// Limitation, stated plainly: this exercises ONE lane at a time. A kernel
// with bar.sync or warp collectives needs real threads and real interop —
// that is not covered here and not claimed.

extern void vadd(unsigned long a, unsigned long b, unsigned long c,
                 unsigned int n, const void *launch);

// One block of 32 lanes computes exactly 32 elements (the kernel's grid
// stride is ntid.x), so the checked size must match the block size.
#define N 32

static float A[N];
static float B[N];
static float C[N];
// 80 bytes: 12 special registers (0..44), scratchpad (48), barrier (56),
// exchange (64).
static unsigned int launch[20];

void _start(void) {
  for (int i = 0; i < N; ++i) {
    A[i] = (float)(i * 3 + 1);
    B[i] = (float)(i - 7);
    C[i] = 0.0f;
  }
  launch[0] = 0;  // tid.x  (per lane, set below)
  launch[3] = 0;  // ctaid.x
  launch[6] = 32; // ntid.x  -> block size
  for (unsigned lane = 0; lane < 32; ++lane) {
    launch[0] = lane; // tid.x
    vadd((unsigned long)A, (unsigned long)B, (unsigned long)C, N, launch);
  }
  int bad = 0;
  for (int i = 0; i < N; ++i) {
    if (C[i] != A[i] + B[i]) ++bad;
  }
  // Linux exit(status): a7 = 93 (SYS_exit), a0 = status.
  register long a0 __asm__("a0") = bad ? 1 : 0;
  register long a7 __asm__("a7") = 93;
  __asm__ volatile("ecall" : : "r"(a0), "r"(a7) : "memory");
  for (;;) {
  }
}
