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
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <pthread.h>

#if defined(__linux__) && defined(__x86_64__)
#include <sys/mman.h>
#endif

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
//   64   : i64 exchange base (per-block warp-exchange region, see below)
namespace {
struct LaunchDesc {
  int32_t spec[12];
  uint64_t smem;      // 48
  uint64_t barrier;   // 56
  uint64_t exchange;  // 64
};
// Warp-exchange slot: one 64-byte slot per thread in the block, indexed by
// linear tid (tid.x + tid.y*ntid.x + tid.z*ntid.x*ntid.y).
//   +0 : i64 row address published by this lane (ldmatrix/stmatrix)
//   +8 : up to 14 x i32 fragment words published by this lane (mma A/B/C)
// The region is per-block scratch memory owned by the driver (one allocation
// per block, threads*64 bytes); a warp-collective op publishes, executes a
// block barrier (stronger than warp scope, still correct), then reads the
// peers' slots. Reused sequentially by every collective in the kernel.
static constexpr size_t kExchangeSlot = 64;
static constexpr size_t kExchangeAddrOff = 0;
static constexpr size_t kExchangeFragOff = 8;
}  // namespace

extern "C" int rouge_kernel_info_query(rouge_kernel_query_fn q,
                                       RougeKernelInfo* info) {
  if (!q || !info) return 0;
  std::memset(info, 0, sizeof(*info));
  return q(info) != 0 ? 1 : 0;
}

extern "C" void* rouge_smem_alloc(const RougeKernelInfo* info) {
  if (!info || info->smem_size == 0) return nullptr;
#if defined(__linux__) && defined(__x86_64__)
  // PTX address math on symbol addresses is 32-bit (mov.u32/add.s32), and
  // the lowering keeps only the low 32 bits in those registers — exactly
  // like hardware, where the shared window fits in 32 bits. A heap scratchpad
  // would lose its high bits in that roundtrip (matrixMul segfaulted here),
  // so the scratchpad lives in low memory: base + any in-bounds offset never
  // crosses 4GB and the trunc/zext roundtrip is bit-exact. The 16 bytes
  // before the aligned user pointer hold {mapping, length} for free().
  const size_t need = static_cast<size_t>(info->smem_size) + 64;
  void* map = mmap(nullptr, need, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_32BIT, -1, 0);
  if (map == MAP_FAILED) return nullptr;  // callers already fail cleanly
  const uintptr_t user =
      (reinterpret_cast<uintptr_t>(map) + 16 + 15) & ~uintptr_t(15);
  static_cast<void**>(reinterpret_cast<void*>(user))[-2] = map;
  *reinterpret_cast<size_t*>(user - 8) = need;
  void* p = reinterpret_cast<void*>(user);
  std::memset(p, 0, info->smem_size);
  return p;
#else
  return std::calloc(1, info->smem_size);
#endif
}

extern "C" void rouge_smem_free(void* p) {
  if (!p) return;
#if defined(__linux__) && defined(__x86_64__)
  void* map = static_cast<void**>(p)[-2];
  const size_t len = *reinterpret_cast<size_t*>(static_cast<char*>(p) - 8);
  if (!map || len < 64) return;  // not ours: leak instead of unmapping garbage
  munmap(map, len);
#else
  std::free(p);
#endif
}

extern "C" uint32_t rouge_smem_offset(const RougeKernelInfo* info,
                                      const char* name) {
  if (!info || !name || !info->shared_vars) return UINT32_MAX;
  for (uint32_t i = 0; i < info->shared_count; ++i) {
    const RougeSharedVar& v = info->shared_vars[i];
    if (v.name && std::strcmp(v.name, name) == 0) return v.offset;
  }
  return UINT32_MAX;
}

extern "C" void* rouge_exchange_alloc(uint32_t threads) {
  if (threads == 0) return nullptr;
  return std::calloc(static_cast<size_t>(threads), kExchangeSlot);
}

extern "C" void rouge_exchange_free(void* p) { std::free(p); }

// bar.sync 0 -> block-wide barrier. Blocks until all threads of the block have
// arrived. A null barrier (kernels without bar.sync) is a no-op.
extern "C" void __rouge_syncthreads(const void* launch) {
  const auto* ld = static_cast<const LaunchDesc*>(launch);
  if (!ld) return;
  auto* b = reinterpret_cast<pthread_barrier_t*>(ld->barrier);
  if (b) pthread_barrier_wait(b);
}
