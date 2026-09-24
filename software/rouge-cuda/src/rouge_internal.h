#pragma once

#include "ptx.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace rouge {

struct Context {
  struct FuncInfo {
    uint64_t moduleKey = 0;
    int fnIndex = 0;
  };

  std::unordered_set<uintptr_t> allocations;          // device ptrs = host addrs
  std::unordered_map<uint64_t, std::shared_ptr<PtxProgram>> modules;  // key = module address
  std::unordered_map<uint64_t, FuncInfo> functions;   // key = function address
};

Context& ctx();

// dlsym-based forwarding: returns a real CUDA symbol if a real library is loaded,
// or nullptr when the CPU-emulated backend should handle the call.
void* load_real_symbol(const char* name);

}  // namespace rouge