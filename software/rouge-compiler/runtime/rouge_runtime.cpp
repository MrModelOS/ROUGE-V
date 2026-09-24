// ROUGE-V runtime — host helpers for AOT-compiled kernels.
//
// This is the seed of the C-API runtime that will replace libcudart/libcuda
// (launch descriptors, shared-memory scratchpads, block barriers). On the
// RISC-V target these functions compile down to fence / custom inter-core
// barrier instructions; on the host test bed they map to pthread primitives.
//
// The kernel-side metadata (__rouge_<kernel>_query) is emitted by ptx2ir from
// the PTX .shared declarations — see runtime/rouge_runtime.h for the ABI.

#include "rouge_runtime.h"

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <pthread.h>

// The IR emitted by ptx2ir fills these structures by hard-coded byte offsets;
// refuse to build on a host where the natural layout disagrees.
static_assert(offsetof(RougeKernelInfo, smem_size) == 0, "ABI: smem_size@0");
static_assert(offsetof(RougeKernelInfo, shared_count) == 4, "ABI: shared_count@4");
static_assert(offsetof(RougeKernelInfo, shared_vars) == 8, "ABI: shared_vars@8");
static_assert(offsetof(RougeSharedVar, name) == 0, "ABI: name@0");
static_assert(offsetof(RougeSharedVar, offset) == 8, "ABI: offset@8");
static_assert(offsetof(RougeSharedVar, size) == 12, "ABI: size@12");
static_assert(offsetof(RougeSharedVar, align) == 16, "ABI: align@16");
static_assert(offsetof(RougeSharedVar, elem_bytes) == 20, "ABI: elem_bytes@20");
static_assert(offsetof(RougeSharedVar, count) == 24, "ABI: count@24");

// Launch descriptor layout (must match ptx2ir / test drivers):
//   0..44: 12 x i32 special registers (tid/ctaid/ntid/nctaid xyz)
//   48   : i64 scratchpad base (block .shared memory)
//   56   : i64 barrier handle
namespace {
struct LaunchDesc {
  int32_t spec[12];
  uint64_t smem;      // 48
  uint64_t barrier;   // 56
};
}  // namespace

extern "C" int rouge_kernel_info_query(rouge_kernel_query_fn q,
                                       RougeKernelInfo* info) {
  if (!q || !info) return 0;
  std::memset(info, 0, sizeof(*info));
  return q(info) != 0 ? 1 : 0;
}

extern "C" void* rouge_smem_alloc(const RougeKernelInfo* info) {
  if (!info || info->smem_size == 0) return nullptr;
  return std::calloc(1, info->smem_size);
}

extern "C" void rouge_smem_free(void* p) { std::free(p); }

extern "C" uint32_t rouge_smem_offset(const RougeKernelInfo* info,
                                      const char* name) {
  if (!info || !name || !info->shared_vars) return UINT32_MAX;
  for (uint32_t i = 0; i < info->shared_count; ++i) {
    const RougeSharedVar& v = info->shared_vars[i];
    if (v.name && std::strcmp(v.name, name) == 0) return v.offset;
  }
  return UINT32_MAX;
}

// bar.sync 0 -> block-wide barrier. Blocks until all threads of the block have
// arrived. A null barrier (kernels without bar.sync) is a no-op.
extern "C" void __rouge_syncthreads(const void* launch) {
  const auto* ld = static_cast<const LaunchDesc*>(launch);
  if (!ld) return;
  auto* b = reinterpret_cast<pthread_barrier_t*>(ld->barrier);
  if (b) pthread_barrier_wait(b);
}
