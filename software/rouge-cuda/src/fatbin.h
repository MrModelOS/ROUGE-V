#pragma once

#include <string>
#include <vector>

// Extracts every PTX module embedded in the nvFatbin container that nvcc links
// into an object file. src/fatbin.cpp documents the layout and how it was
// verified. Returns the PTX texts; on failure returns nothing and fills *err
// with a human-readable reason (never a silent empty result).
std::vector<std::string> rouge_fatbin_extract_ptx(const void* fatCubinHandle,
                                                   std::string* err);