// ROUGE-V host runtime — public C API for AOT-compiled kernels.
//
// The runtime is the seed of the libcudart/libcuda replacement. It owns the
// launch descriptor (see below), the per-block .shared scratchpad and the
// block barrier. AOT kernels compiled by ptx2ir carry their own .shared
// layout in the emitted IR; every translated kernel therefore exports
//
//   i32 __rouge_<kernel>_query(RougeKernelInfo* info)
//
// which fills this structure, so the driver never hard-codes scratchpad
// sizes or symbol offsets (they come from the PTX itself).
//
// Launch descriptor layout (must match ptx2ir / test drivers):
//   0..44: 12 x i32 special registers (tid/ctaid/ntid/nctaid xyz)
//   48   : i64 scratchpad base (block .shared memory)
//   56   : i64 barrier handle

#ifndef ROUGE_RUNTIME_H_
#define ROUGE_RUNTIME_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// One .shared variable as laid out by the PTX parser (alignment-padded byte
// offsets; see docs/06-compiler-architecture.md).
typedef struct RougeSharedVar {
  const char* name;       // PTX symbol, e.g. "smem"
  uint32_t offset;        // byte offset inside the block scratchpad
  uint32_t size;          // total bytes (elem_bytes * count)
  uint32_t align;         // declared alignment in bytes
  uint32_t elem_bytes;    // 1, 2, 4 or 8
  uint32_t count;         // number of elements
  uint32_t reserved;      // padding -> struct size is 32 bytes
} RougeSharedVar;

// Kernel metadata filled by __rouge_<kernel>_query.
// Field offsets (0/4/8) are part of the ABI emitted by ptx2ir; the runtime
// static_asserts the layout on the host.
typedef struct RougeKernelInfo {
  uint32_t smem_size;      // offset 0: scratchpad bytes the block requires
  uint32_t shared_count;   // offset 4: number of .shared variables
  RougeSharedVar* shared_vars;  // offset 8: RougeSharedVar[shared_count]
} RougeKernelInfo;

typedef int32_t (*rouge_kernel_query_fn)(RougeKernelInfo* info);

// Query a translated kernel's shared-memory layout. Returns 1 on success.
int rouge_kernel_info_query(rouge_kernel_query_fn q, RougeKernelInfo* info);

// Allocate / release a zeroed per-block scratchpad for the queried kernel.
void* rouge_smem_alloc(const RougeKernelInfo* info);
void rouge_smem_free(void* p);

// Byte offset of a .shared symbol, or UINT32_MAX when the kernel has no such
// variable.
uint32_t rouge_smem_offset(const RougeKernelInfo* info, const char* name);

// bar.sync 0 -> block-wide barrier (pthread barrier on the host, fence on
// RISC-V). A null launch descriptor or a null barrier handle is a no-op.
void __rouge_syncthreads(const void* launch);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // ROUGE_RUNTIME_H_
