#pragma once

#include "ptx.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace rouge {

struct Context {
  struct FuncInfo {
    uint64_t moduleKey = 0;
    int fnIndex = 0;
  };

  std::unordered_set<uintptr_t> allocations;          // device ptrs = host addrs
  std::unordered_map<uint64_t, std::shared_ptr<PtxProgram>> modules;  // key = module address
  std::unordered_map<uint64_t, FuncInfo> functions;   // key = function address
  // Module-global backing storage: key = mixed (moduleKey, globalIndex),
  // zero-initialised like CUDA device memory. Owned here, freed on
  // cuModuleUnload; the same bytes are handed to cuModuleGetGlobal callers
  // and (by address) to the interpreter, so host writes are kernel-visible.
  std::unordered_map<uint64_t, std::vector<uint8_t>> globalStorage;
};

inline uint64_t global_storage_key(uint64_t moduleKey, int globalIndex) {
  return moduleKey + static_cast<uint64_t>(globalIndex) * 0x9E3779B97F4A7C15ull;
}

Context& ctx();

// Runtime-path module registry: PTX modules from ROUGE_CUDA_PTX and from
// fatbin images embedded in foreign binaries. Distinct from ctx().modules
// (Driver API cuModuleLoadData); kernel lookup searches both, runtime first.
inline std::vector<std::shared_ptr<PtxProgram>>& runtime_modules() {
  static std::vector<std::shared_ptr<PtxProgram>> mods;
  return mods;
}

// Registered kernel stubs: stub address -> mangled kernel name, filled by
// __cudaRegisterFunction. Header-inline singleton so runtime_register.cpp,
// runtime_api.cpp and driver_api.cpp share one table.
inline std::unordered_map<uintptr_t, std::string>& stub_names() {
  static std::unordered_map<uintptr_t, std::string> table;
  return table;
}

// dlsym-based forwarding: returns a real CUDA symbol if a real library is loaded,
// or nullptr when the CPU-emulated backend should handle the call.
void* load_real_symbol(const char* name);

}  // namespace rouge