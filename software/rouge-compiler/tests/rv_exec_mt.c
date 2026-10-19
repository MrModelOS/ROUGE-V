// ROUGE-V: freestanding RISC-V harness with REAL threads.
//
// The single-lane harness (rv_exec_harness.c) covers kernels without
// barriers. This one covers the harder case: a block kernel with bar.sync,
// executed by 32 real pthreads on RISC-V (rv64gcv) under qemu-user.
//
// It uses the project's own runtime (rouge_runtime.cpp): the kernel's
// .shared layout comes from __rouge_block_reduce_query, the scratchpad from
// rouge_smem_alloc, and bar.sync from __rouge_syncthreads (a pthread barrier).
// So the barrier path of the RISC-V build is exercised for real, not
// simulated.
//
// Build: needs a riscv64 sysroot (headers + libc.a + libgcc) and lld; the
// CI/test script SKIPs honestly when they are absent.

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---- the project's own runtime (rouge_runtime.h) --------------------------
typedef struct RougeSharedVar {
  const char *name;
  uint32_t offset;
  uint32_t size;
  uint32_t align;
  uint32_t elem_bytes;
  uint32_t count;
  uint32_t reserved;
} RougeSharedVar;

typedef struct RougeKernelInfo {
  uint32_t smem_size;
  uint32_t shared_count;
  RougeSharedVar *shared_vars;
} RougeKernelInfo;

typedef int32_t (*rouge_kernel_query_fn)(RougeKernelInfo *info);
int rouge_kernel_info_query(rouge_kernel_query_fn q, RougeKernelInfo *info);
void *rouge_smem_alloc(const RougeKernelInfo *info);
void rouge_smem_free(void *p);
uint32_t rouge_smem_offset(const RougeKernelInfo *info, const char *name);

// ptx2ir signature:
//   define void @block_reduce(i64 %data, i64 %out, i32 %n, ptr %launch)
extern void block_reduce(uint64_t data, uint64_t out, uint32_t n,
                          const void *launch);
extern int32_t __rouge_block_reduce_query(RougeKernelInfo *info);

struct Launch {
  int32_t tidx, tiy, tiz;
  int32_t ctaidx, ctaidy, ctaidz;
  int32_t ntidx, ntidy, ntidz;
  int32_t nctaidx, nctaidy, nctaidz;
  uint64_t smem;  // offset 48: scratchpad
  void *barrier;  // offset 56: block barrier
};

#define BX 32
#define GX 4
#define N (BX * GX)

struct Args {
  float *data;
  float *out;
  pthread_barrier_t *barrier;
  void *scratch;
  int32_t ctaidx;
  int32_t tidx;  // lane index of THIS thread
};

static void *worker(void *p) {
  struct Args *a = (struct Args *)p;
  // One thread == one lane: the kernel's block has BX threads, and the
  // barrier inside it expects exactly BX participants.
  struct Launch L;
  memset(&L, 0, sizeof(L));
  L.tidx = a->tidx;
  L.ctaidx = a->ctaidx;
  L.ntidx = BX;
  L.nctaidx = GX;
  L.smem = (uint64_t)(uintptr_t)a->scratch;
  L.barrier = a->barrier;
  block_reduce((uint64_t)(uintptr_t)a->data, (uint64_t)(uintptr_t)a->out, N,
               &L);
  return 0;
}

int main(void) {
  RougeKernelInfo info;
  memset(&info, 0, sizeof(info));
  if (!rouge_kernel_info_query(&__rouge_block_reduce_query, &info)) {
    printf("rv mt: FAIL - kernel query returned 0\n");
    return 1;
  }
  if (info.smem_size == 0 || info.shared_count != 1) {
    printf("rv mt: FAIL - bad kernel info (smem=%u vars=%u)\n", info.smem_size,
           info.shared_count);
    return 1;
  }
  if (rouge_smem_offset(&info, "smem") == UINT32_MAX) {
    printf("rv mt: FAIL - no 'smem' in the shared layout\n");
    return 1;
  }
  float *data = (float *)calloc(N, sizeof(float));
  float *out = (float *)calloc(GX, sizeof(float));
  if (!data || !out) return 1;
  for (int i = 0; i < N; ++i) data[i] = (float)(i + 1);

  pthread_barrier_t *barriers =
      (pthread_barrier_t *)calloc(GX, sizeof(pthread_barrier_t));
  void **scratch = (void **)calloc(GX, sizeof(void *));
  if (!barriers || !scratch) return 1;

  pthread_t *threads = (pthread_t *)calloc((size_t)BX * GX, sizeof(pthread_t));
  struct Args *args = (struct Args *)calloc((size_t)GX, sizeof(struct Args));
  if (!threads || !args) return 1;

  for (int32_t b = 0; b < GX; ++b) {
    pthread_barrier_init(&barriers[b], 0, BX);
    scratch[b] = rouge_smem_alloc(&info);
    if (!scratch[b]) {
      printf("rv mt: FAIL - scratchpad alloc\n");
      return 1;
    }
    args[b].data = data;
    args[b].out = out;
    args[b].barrier = &barriers[b];
    args[b].scratch = scratch[b];
    args[b].ctaidx = b;
    for (int32_t t = 0; t < BX; ++t) {
      struct Args *ta = (struct Args *)calloc(1, sizeof(struct Args));
      if (!ta) return 1;
      *ta = args[b];
      ta->tidx = t;
      if (pthread_create(&threads[b * BX + t], 0, worker, ta) != 0) {
        printf("rv mt: FAIL - pthread_create\n");
        return 1;
      }
    }
  }
  for (int i = 0; i < BX * GX; ++i) pthread_join(threads[i], 0);

  int bad = 0;
  for (int32_t b = 0; b < GX; ++b) {
    float expected = 0.0f;
    for (int32_t i = b * BX; i < (b + 1) * BX; ++i) expected += data[i];
    if (out[b] != expected) {
      if (bad < 4)
        printf("rv mt: block %d got %.6f want %.6f\n", b, (double)out[b],
               (double)expected);
      ++bad;
    }
  }
  for (int32_t b = 0; b < GX; ++b) {
    pthread_barrier_destroy(&barriers[b]);
    rouge_smem_free(scratch[b]);
  }
  if (bad) {
    printf("rv mt: FAIL - %d mismatches\n", bad);
    return 1;
  }
  printf("rv mt: OK - %d blocks x %d threads with bar.sync on rv64gcv\n", GX,
         BX);
  return 0;
}
