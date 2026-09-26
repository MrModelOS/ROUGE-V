#include "ptx.h"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

namespace rouge {

namespace {

std::string trim(const std::string& s) {
  const size_t b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos) return std::string();
  const size_t e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

std::string strip_comment(const std::string& line) {
  const size_t p = line.find("//");
  if (p == std::string::npos) return line;
  return line.substr(0, p);
}

// "%r1" -> "r1" ; "[%rd1]" -> "[rd1]" ; everything else unchanged.
std::string norm_operand(const std::string& a) {
  if (!a.empty() && a[0] == '%') return a.substr(1);
  if (a.size() >= 3 && a.front() == '[' && a.back() == ']') {
    std::string inner = trim(a.substr(1, a.size() - 2));
    if (!inner.empty() && inner[0] == '%') return "[" + inner.substr(1) + "]";
  }
  return a;
}

bool is_label_line(const std::string& s, std::string* label) {
  if (s.empty()) return false;
  const size_t colon = s.find(':');
  if (colon == std::string::npos) return false;
  for (size_t i = colon + 1; i < s.size(); ++i)
    if (!std::isspace((unsigned char)s[i])) return false;
  const std::string name = s.substr(0, colon);
  if (name.empty() || !std::isalpha((unsigned char)name[0])) return false;
  for (char c : name)
    if (!std::isalnum((unsigned char)c) && c != '_') return false;
  *label = name;
  return true;
}

bool parse_instruction(const std::string& raw, PtxInstruction* out) {
  std::string s = trim(raw);
  if (s.empty()) return true;
  const size_t semi = s.find(';');
  if (semi != std::string::npos) s = s.substr(0, semi);
  s = trim(s);
  if (s.empty()) return true;

  std::string rest = s;
  if (rest[0] == '@') {  // predicate
    const size_t sp = rest.find(' ');
    if (sp == std::string::npos) return false;
    std::string pred = trim(rest.substr(1, sp - 1));
    // Negation may precede the register sigil: "@!%p1" as well as "@%p1" and
    // "@!p1" all occur in real PTX. Strip '!' first, then '%', and keep the
    // '!' prefix so consumers see the canonical "!p1" / "p1" form.
    bool negate = false;
    if (!pred.empty() && pred[0] == '!') {
      negate = true;
      pred = pred.substr(1);
    }
    if (!pred.empty() && pred[0] == '%') pred = pred.substr(1);
    if (negate) pred = "!" + pred;
    out->pred = pred;
    rest = trim(rest.substr(sp + 1));
  }

  const size_t sp = rest.find(' ');
  if (sp == std::string::npos) {
    out->op = rest;
  } else {
    out->op = rest.substr(0, sp);
    const std::string argstr = trim(rest.substr(sp + 1));
    if (!argstr.empty()) {
      std::stringstream ss(argstr);
      std::string tok;
      while (std::getline(ss, tok, ',')) {
        tok = norm_operand(trim(tok));  // trim FIRST, then normalize registers
        if (!tok.empty()) out->args.push_back(tok);
      }
    }
  }
  return true;
}

uint32_t param_width(const std::string& type) {
  if (type == ".u64" || type == ".s64" || type == ".b64" || type == ".f64") return 8;
  if (type == ".u32" || type == ".s32" || type == ".b32" || type == ".f32") return 4;
  return 0;
}

// Parse ".reg .b32 %r<9>;" / ".reg .b64 %rd0, %rd1;" / ".reg .pred %p<2>;"
// declarations and record per-register widths for the compiler front-end.
void parse_reg_decl(const std::string& line, std::unordered_map<std::string, int>* regs) {
  std::string s = trim(line);
  const size_t semi = s.find(';');
  if (semi != std::string::npos) s = s.substr(0, semi);
  std::stringstream ss(s);
  std::string tok;
  ss >> tok;  // ".reg"
  ss >> tok;  // ".b32" / ".b64" / ".pred"
  const std::string type = tok;
  const int bits = (type == ".b64" || type == ".u64" || type == ".s64" || type == ".f64")
                       ? 64
                       : (type == ".pred")
                           ? 1
                           : (type == ".b16" || type == ".u16" || type == ".s16" ||
                              type == ".f16" || type == ".bf16")
                               ? 16
                               : 32;
  while (ss >> tok) {
    std::string t = trim(tok);
    if (!t.empty() && t.back() == ',') t.pop_back();
    if (t.empty() || t[0] != '%') continue;
    const size_t lt = t.find('<');
    if (lt != std::string::npos) {  // range form "%r<9>" -> %r0..%r8
      const size_t gt = t.find('>', lt);
      if (gt == std::string::npos) continue;
      const std::string base = t.substr(1, lt - 1);
      const int count = std::atoi(t.substr(lt + 1, gt - lt - 1).c_str());
      for (int i = 0; i < count; ++i) (*regs)[base + std::to_string(i)] = bits;
    } else {
      (*regs)[t.substr(1)] = bits;
    }
  }
}

int shared_elem_bytes(const std::string& type) {
  if (type == ".b8" || type == ".u8" || type == ".s8") return 1;
  if (type == ".b16" || type == ".u16" || type == ".s16" || type == ".f16") return 2;
  if (type == ".f64" || type == ".b64" || type == ".u64" || type == ".s64") return 8;
  return 4;  // .b32/.u32/.s32/.f32
}

// Parse a module-scope ".global .align 4 .b32 arr[4096];" / ".const ...".
// nvcc emits one of these per __device__ variable; "cvta.to.global.u64 %rd,
// arr" then has to yield the symbol's real address.
void parse_global_decl(const std::string& line, PtxProgram* prog) {
  std::string s = trim(line);
  const size_t semi = s.find(';');
  if (semi != std::string::npos) s = s.substr(0, semi);
  std::stringstream ss(s);
  std::string tok;
  int align = 0, elemBytes = 4;
  bool isConst = false;
  ss >> tok;  // ".global" or ".const"
  if (tok == ".const") isConst = true;
  while (ss >> tok) {
    if (tok == ".align") {
      ss >> tok;
      align = std::atoi(tok.c_str());
    } else if (!tok.empty() && tok[0] == '.') {
      elemBytes = shared_elem_bytes(tok);
    } else {
      std::string t = tok;
      if (!t.empty() && t.back() == ',') t.pop_back();
      if (t.empty()) continue;
      PtxGlobalVar v;
      const size_t lb = t.find('[');
      if (lb != std::string::npos) {
        const size_t rb = t.find(']', lb);
        if (rb != std::string::npos) {
          v.name = t.substr(0, lb);
          v.count = std::atoi(t.substr(lb + 1, rb - lb - 1).c_str());
        }
      } else {
        v.name = t;
      }
      if (v.name.empty()) continue;
      v.elemBytes = elemBytes;
      v.align = align > 0 ? align : (elemBytes >= 8 ? 8 : elemBytes);
      v.isConst = isConst;
      prog->globalIndex[v.name] = static_cast<int>(prog->globals.size());
      prog->globals.push_back(std::move(v));
    }
  }
}

int align_up(int v, int a) { return (v + a - 1) / a * a; }

// Parse one ".param ..." entry of a kernel signature. Pointer/space/align
// attributes are tolerated: the last recognised width token wins (so
// ".param .u32 .ptr .global .b64 name" is stored as 8 bytes), the name is the
// last token. Real nvcc output puts the whole signature on one line, so this
// is fed comma-separated entries by the parser below.
bool parse_param_entry(const std::string& entry, PtxFunction* fn, int lineNo,
                       std::string* error) {
  std::stringstream pss(entry);
  std::string tok, name;
  if (!(pss >> tok) || tok != ".param") return true;  // not a parameter
  uint32_t w = 0;
  while (pss >> tok) {
    name = tok;
    const uint32_t tw = param_width(tok);
    if (tw != 0) w = tw;
  }
  if (w == 0 || name.empty()) {
    if (error)
      *error = "unsupported param type in '" + entry + "' (line " +
               std::to_string(lineNo) + ")";
    return false;
  }
  PtxParam p;
  p.name = name;
  p.width = static_cast<int>(w);
  p.index = static_cast<int>(fn->params.size());
  p.offset = fn->paramsBlobSize;
  fn->paramsBlobSize += static_cast<int>(w);
  fn->params.push_back(std::move(p));
  return true;
}

// Parse ".shared .align 4 .b32 smem[256];" (or a comma list of vars) and lay
// them out in the block scratchpad: successive variables get alignment-padded
// byte offsets; PtxFunction::sharedSize is the total scratchpad size.
void parse_shared_decl(const std::string& line, PtxFunction* fn) {
  std::string s = trim(line);
  const size_t semi = s.find(';');
  if (semi != std::string::npos) s = s.substr(0, semi);
  std::stringstream ss(s);
  std::string tok;
  int align = 0;
  int elemBytes = 4;
  ss >> tok;  // ".shared"
  while (ss >> tok) {
    if (tok == ".align") {
      ss >> tok;
      align = std::atoi(tok.c_str());
    } else if (!tok.empty() && tok[0] == '.') {
      elemBytes = shared_elem_bytes(tok);
    } else {
      std::string t = tok;
      if (!t.empty() && t.back() == ',') t.pop_back();
      if (t.empty()) continue;
      PtxSharedVar v;
      const size_t lb = t.find('[');
      if (lb != std::string::npos) {
        const size_t rb = t.find(']', lb);
        if (rb != std::string::npos) {
          v.name = t.substr(0, lb);
          v.count = std::atoi(t.substr(lb + 1, rb - lb - 1).c_str());
        }
      }
      if (v.name.empty()) continue;
      v.elemBytes = elemBytes;
      v.align = align > 0 ? align : (elemBytes >= 8 ? 8 : elemBytes);
      v.offset = align_up(fn->sharedSize, v.align);
      fn->sharedSize = v.offset + v.elemBytes * v.count;
      fn->sharedIndex[v.name] = static_cast<int>(fn->sharedVars.size());
      fn->sharedVars.push_back(std::move(v));
    }
  }
  fn->sharedSize = align_up(fn->sharedSize, 16);  // scratchpad base alignment
}

template <typename To, typename From>
To bit_cast_v(const From& src) {
  static_assert(sizeof(To) == sizeof(From), "bit_cast_v: size mismatch");
  To dst;
  std::memcpy(&dst, &src, sizeof(To));
  return dst;
}

// ---- f16 / bf16 bit-level conversions (IEEE binary16 / bfloat16) ---------
// Implemented in software so the interpreter needs no <f16.h>/BF16 hardware
// and produces bit-identical results on every host. Rounding is
// round-to-nearest-even, matching LLVM fptrunc and the CUDA fp16/bf16 rules.

float f16_to_f32(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1fu;
  const uint32_t man = h & 0x3ffu;
  uint32_t bits;
  if (exp == 0) {
    if (man == 0) {
      bits = sign;  // +-0
    } else {
      // Subnormal binary16: normalize, then rebias.
      uint32_t m = man;
      uint32_t shift = 0;
      while ((m & 0x400u) == 0) { m <<= 1; ++shift; }
      m &= 0x3ffu;
      bits = sign | ((113u - shift) << 23) | (m << 13);
    }
  } else if (exp == 0x1fu) {
    bits = sign | 0x7f800000u | (man << 13);  // inf / NaN
  } else {
    bits = sign | ((exp - 15u + 127u) << 23) | (man << 13);
  }
  return bit_cast_v<float>(bits);
}

uint16_t f32_to_f16_rne(float f) {
  const uint32_t x = bit_cast_v<uint32_t>(f);
  const uint16_t sign = static_cast<uint16_t>((x >> 16) & 0x8000u);
  const uint32_t e = (x >> 23) & 0xffu;
  const uint32_t man = x & 0x7fffffu;
  if (e == 0xffu) {  // inf / NaN
    return man ? static_cast<uint16_t>(sign | 0x7e00u)  // quiet NaN
               : static_cast<uint16_t>(sign | 0x7c00u);
  }
  const int exp = static_cast<int>(e) - 127 + 15;
  if (exp >= 0x1f) return static_cast<uint16_t>(sign | 0x7c00u);  // overflow
  if (exp <= 0) {                                                   // subnormal
    if (exp < -10) return sign;                                    // underflow
    const uint32_t m = man | 0x800000u;                            // implicit 1
    const uint32_t shift = static_cast<uint32_t>(14 - exp);
    uint32_t val = m >> shift;
    const uint32_t rem = m & ((1u << shift) - 1u);
    const uint32_t half = 1u << (shift - 1);
    if (rem > half || (rem == half && (val & 1u))) ++val;
    return static_cast<uint16_t>(sign | val);
  }
  uint32_t val = (static_cast<uint32_t>(exp) << 10) | (man >> 13);
  const uint32_t rem = man & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (val & 1u))) ++val;  // may carry to inf
  return static_cast<uint16_t>(sign | val);
}

float bf16_to_f32(uint16_t b) {
  return bit_cast_v<float>(static_cast<uint32_t>(b) << 16);
}

// bfloat16 = top 16 bits of binary32 with RNE (0x7fff + lsb tie-break).
uint16_t f32_to_bf16_rne(float f) {
  const uint32_t x = bit_cast_v<uint32_t>(f);
  if (((x >> 23) & 0xffu) == 0xffu)  // inf / NaN pass through
    return static_cast<uint16_t>((x >> 16) & 0xffffu);
  const uint32_t lsb = (x >> 16) & 1u;
  return static_cast<uint16_t>((x + 0x7fffu + lsb) >> 16);
}

}  // namespace

// ---------------------------------------------------------------- parser

std::unique_ptr<PtxProgram> parse_ptx(const std::string& text, std::string* error) {
  auto prog = std::make_unique<PtxProgram>();
  std::stringstream ss(text);
  std::string line;
  PtxFunction* cur = nullptr;
  bool inParams = false;
  int lineNo = 0;

  while (std::getline(ss, line)) {
    ++lineNo;
    const std::string s = trim(strip_comment(line));
    if (s.empty()) continue;

    // Function header: ".visible .entry name(" or ".entry name("
    if (s.rfind(".visible .entry", 0) == 0 || s.rfind(".entry ", 0) == 0) {
      const size_t epos = s.find(".entry") + 7;  // past ".entry"
      std::string rest = trim(s.substr(epos));
      const size_t paren = rest.find('(');
      const std::string fname = trim(paren == std::string::npos ? rest : rest.substr(0, paren));
      prog->functions.push_back(PtxFunction{});
      cur = &prog->functions.back();
      cur->name = fname;
      prog->functionIndex[fname] = static_cast<int>(prog->functions.size()) - 1;
      if (paren == std::string::npos) {
        inParams = false;
      } else {
        // Real nvcc output keeps the whole signature on ONE line:
        //   .visible .entry k(.param .u64 a, .param .u32 n) {
        // while hand-written PTX spreads it over several lines. Support both.
        const size_t close = rest.find(')', paren);
        if (close == std::string::npos) {
          inParams = true;  // multi-line signature
        } else {
          std::stringstream sig(rest.substr(paren + 1, close - paren - 1));
          std::string entry;
          while (std::getline(sig, entry, ',')) {
            entry = trim(entry);
            if (entry.empty()) continue;
            if (!parse_param_entry(entry, cur, lineNo, error)) return nullptr;
          }
          inParams = false;
        }
      }
      continue;
    }

    if (!cur) {  // module-scope: .version/.target/.address_size and .global
      if (s.rfind(".global", 0) == 0 || s.rfind(".const", 0) == 0)
        parse_global_decl(s, prog.get());
      continue;
    }

    if (inParams) {
      if (s == ")") {
        inParams = false;
        continue;
      }
      if (s.rfind(".param", 0) == 0) {
        // Multi-line signature: one or more comma-separated entries, possibly
        // with the closing ")" on the same line.
        std::stringstream pss(s);
        std::string entry;
        while (std::getline(pss, entry, ',')) {
          entry = trim(entry);
          const size_t cp = entry.find(')');
          if (cp != std::string::npos) {
            entry = trim(entry.substr(0, cp));
            inParams = false;
          }
          if (entry.empty()) continue;
          if (!parse_param_entry(entry, cur, lineNo, error)) return nullptr;
        }
      }
      continue;
    }

    // Function body.
    if (s == "{") continue;
    if (s == "}") continue;
    if (s.rfind(".reg", 0) == 0) {
      parse_reg_decl(s, &cur->regBits);
      continue;
    }
    if (s.rfind(".shared", 0) == 0) {
      parse_shared_decl(s, cur);
      continue;
    }
    if (!s.empty() && s[0] == '.') continue;  // .align/... directives

    std::string label;
    if (is_label_line(s, &label)) {
      cur->labels[label] = static_cast<int>(cur->body.size());
      continue;
    }

    PtxInstruction instr;
    if (!parse_instruction(s, &instr)) {
      if (error) *error = "cannot parse instruction (line " + std::to_string(lineNo) + "): " + s;
      return nullptr;
    }
    cur->body.push_back(std::move(instr));
  }

  return prog;
}

// ------------------------------------------------------------ interpreter

namespace {

struct ExecState {
  std::unordered_map<std::string, uint64_t> regs;
  std::unordered_map<std::string, bool> preds;
  int pc = 0;
  const PtxFunction* fn = nullptr;
  const uint8_t* params = nullptr;
  uintptr_t sharedBase = 0;  // host address of this block's scratchpad
  uint32_t tid[3] = {0, 0, 0};
  uint32_t ctaid[3] = {0, 0, 0};
  uint32_t ntid[3] = {1, 1, 1};
  uint32_t nctaid[3] = {1, 1, 1};

  uint64_t reg(const std::string& n) const {
    const auto it = regs.find(n);
    return it == regs.end() ? 0 : it->second;
  }
  void setreg(const std::string& n, uint64_t v) { regs[n] = v; }

  bool spec_reg(const std::string& n, uint64_t* out) const {
    if (n == "tid.x") { *out = tid[0]; return true; }
    if (n == "tid.y") { *out = tid[1]; return true; }
    if (n == "tid.z") { *out = tid[2]; return true; }
    if (n == "ctaid.x") { *out = ctaid[0]; return true; }
    if (n == "ctaid.y") { *out = ctaid[1]; return true; }
    if (n == "ctaid.z") { *out = ctaid[2]; return true; }
    if (n == "ntid.x") { *out = ntid[0]; return true; }
    if (n == "ntid.y") { *out = ntid[1]; return true; }
    if (n == "ntid.z") { *out = ntid[2]; return true; }
    if (n == "nctaid.x") { *out = nctaid[0]; return true; }
    if (n == "nctaid.y") { *out = nctaid[1]; return true; }
    if (n == "nctaid.z") { *out = nctaid[2]; return true; }
    return false;
  }
};

uint64_t read_mem(const uint8_t* p, int width) {
  uint64_t v = 0;
  std::memcpy(&v, p, static_cast<size_t>(width));
  return v;
}

// Read an operand that may be a special register, an immediate literal, or a GPR.
uint64_t operand_u64(const ExecState& st, const std::string& tok) {
  uint64_t v = 0;
  if (st.spec_reg(tok, &v)) return v;
  if (!tok.empty() && (std::isdigit(static_cast<unsigned char>(tok[0])) || tok[0] == '-'))
    return static_cast<uint64_t>(std::strtoull(tok.c_str(), nullptr, 10));
  return st.reg(tok);
}

std::string strip_brackets(const std::string& a) {
  if (a.size() >= 3 && a.front() == '[' && a.back() == ']')
    return a.substr(1, a.size() - 2);
  return a;
}

// If the operand (e.g. "[smem]" or "smem") names a .shared variable, return its
// host scratchpad address. Shared symbols are addressed as base + laid-out
// offset (identity mapping, same as the AOT path).
bool shared_symbol_addr(const PtxFunction& fn, const ExecState& st,
                        const std::string& tok, uintptr_t* out) {
  const std::string inner = strip_brackets(tok);
  const auto it = fn.sharedIndex.find(inner);
  if (it == fn.sharedIndex.end()) return false;
  *out = static_cast<uintptr_t>(st.sharedBase) +
         static_cast<uintptr_t>(fn.sharedVars[it->second].offset);
  return true;
}

// Split a PTX memory operand into a base symbol/register and a constant byte
// offset: "[%r6]" -> ("r6", 0), "[%r6+512]" -> ("r6", 512), "[smem-4]" -> ...
// Real nvcc output indexes shared memory exactly this way, so the offset form
// is not optional. Returns false when there is no trailing +/-Const.
bool split_reg_offset(const std::string& inner, std::string* base,
                      int64_t* offset) {
  const size_t p = inner.find_first_of("+-");
  if (p == std::string::npos || p == 0) return false;
  const std::string tail = inner.substr(p + 1);
  if (tail.empty()) return false;
  for (char c : tail)
    if (!std::isdigit(static_cast<unsigned char>(c))) return false;
  *base = inner.substr(0, p);
  *offset = std::strtoll(tail.c_str(), nullptr, 10);
  return true;
}

// Resolve a memory operand: shared symbol first, then a shared symbol with a
// constant offset, then a GPR holding an address (global, or a shared address
// produced by cvta.to.shared), then a GPR plus a constant offset.
uintptr_t mem_addr(const PtxFunction& fn, const ExecState& st,
                   const std::string& tok) {
  uintptr_t shared = 0;
  if (shared_symbol_addr(fn, st, tok, &shared)) return shared;
  const std::string inner = strip_brackets(tok);
  std::string base;
  int64_t off = 0;
  if (split_reg_offset(inner, &base, &off))
    return static_cast<uintptr_t>(st.reg(base) + static_cast<uint64_t>(off));
  return static_cast<uintptr_t>(st.reg(inner));
}

const PtxParam* find_param(const PtxFunction& fn, const std::string& name) {
  for (const auto& p : fn.params)
    if (p.name == name) return &p;
  return nullptr;
}

// ---- integer ALU helpers -------------------------------------------------
// Everything below is written with plain C++ operators and explicit masks so
// the interpreter lands on exactly the bits the AOT path gets from LLVM's
// integer instructions (same operands, same widths, same shifts). None of it
// relies on implementation-defined behaviour (signed overflow, >> on a
// negative value, INT_MIN / -1, out-of-range float->int casts).

bool ends_with(const std::string& s, const char* suffix) {
  const size_t n = std::strlen(suffix);
  return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

// Width in bits of the integer type a mnemonic ends with (".s32", ".u64",
// ".b16", ...); 0 for float / predicate / vector forms so callers can route
// those to the dedicated blocks.
int int_type_bits(const std::string& op) {
  if (ends_with(op, ".s64") || ends_with(op, ".u64") || ends_with(op, ".b64")) return 64;
  if (ends_with(op, ".s32") || ends_with(op, ".u32") || ends_with(op, ".b32")) return 32;
  if (ends_with(op, ".s16") || ends_with(op, ".u16") || ends_with(op, ".b16")) return 16;
  return 0;
}

bool is_signed_int_type(const std::string& op) {
  return ends_with(op, ".s16") || ends_with(op, ".s32") || ends_with(op, ".s64");
}

// Truncate a computed value to the width of the destination register, so stale
// upper bits never leak into a later read of the same register.
uint64_t width_mask(uint64_t v, int bits) {
  if (bits >= 64) return v;
  return v & ((1ull << bits) - 1ull);
}

// Sign-extend the low `bits` of v to 64 bits (no UB: v is unsigned).
uint64_t sext64(uint64_t v, int bits) {
  if (bits >= 64) return v;
  const uint64_t sign = 1ull << (bits - 1);
  return (v ^ sign) - sign;
}

// Arithmetic shift right (PTX shr.s32 / shr.s64): shifts in copies of the
// sign bit without relying on the pre-C++20 rule for negative operands.
uint32_t sar32(uint32_t v, unsigned sh) {
  if (sh == 0) return v;
  if ((v & 0x80000000u) == 0) return v >> sh;
  return ~((~v) >> sh);
}

uint64_t sar64(uint64_t v, unsigned sh) {
  if (sh == 0) return v;
  if ((v & 0x8000000000000000ull) == 0) return v >> sh;
  return ~((~v) >> sh);
}

unsigned popcount32(uint32_t v) {
  v = v - ((v >> 1) & 0x55555555u);
  v = (v & 0x33333333u) + ((v >> 2) & 0x33333333u);
  v = (v + (v >> 4)) & 0x0f0f0f0fu;
  return (v * 0x01010101u) >> 24;
}

unsigned popcount64(uint64_t v) {
  return popcount32(static_cast<uint32_t>(v)) +
         popcount32(static_cast<uint32_t>(v >> 32));
}

unsigned clz32(uint32_t v) {
  if (v == 0) return 32;
  unsigned n = 0;
  if ((v & 0xffff0000u) == 0) { n += 16; v <<= 16; }
  if ((v & 0xff000000u) == 0) { n += 8;  v <<= 8;  }
  if ((v & 0xf0000000u) == 0) { n += 4;  v <<= 4;  }
  if ((v & 0xc0000000u) == 0) { n += 2;  v <<= 2;  }
  if ((v & 0x80000000u) == 0) { n += 1; }
  return n;
}

unsigned clz64(uint64_t v) {
  if (v == 0) return 64;
  if (static_cast<uint32_t>(v >> 32) != 0) return clz32(static_cast<uint32_t>(v >> 32));
  return 32u + clz32(static_cast<uint32_t>(v));
}

// PTX float/double -> integer conversion: round toward zero and saturate at
// the destination range (NaN -> 0), which is what the hardware and the AOT
// path's fptosi/fptoui produce. Casts are only reached for in-range values.
uint64_t fp_to_int(double d, int bits, bool sgn) {
  if (std::isnan(d)) return 0;
  const uint64_t half =
      (bits >= 64) ? 0x8000000000000000ull : (1ull << (bits - 1));
  const uint64_t umax = (bits >= 64) ? ~0ull : ((1ull << bits) - 1ull);
  if (sgn) {
    const double lo = -std::ldexp(1.0, bits - 1);
    const double hi = std::ldexp(1.0, bits - 1);
    if (d <= lo) return half;
    if (d >= hi) return half - 1ull;
    return static_cast<uint64_t>(static_cast<int64_t>(d));
  }
  if (d <= 0.0) return 0;
  const double hi = std::ldexp(1.0, bits);
  if (d >= hi) return umax;
  return static_cast<uint64_t>(d);
}

// Operands are split on ',', so a vector destination "{%r2, %r3}" reaches the
// interpreter as the two tokens "{%r2" and "r3}". Strip the braces and the
// "%" prefix (norm_operand only drops a leading one) to get the register names.
std::string vector_reg_name(const std::string& tok) {
  std::string s = trim(tok);
  if (!s.empty() && s.front() == '{') s.erase(s.begin());
  if (!s.empty() && s.back() == '}') s.pop_back();
  s = trim(s);
  if (!s.empty() && s.front() == '%') s.erase(s.begin());
  return s;
}

// Warp state for shfl/vote fallbacks: the CTA scheduler sets this to the
// current block's states[] before stepping, so run_thread_step can read a
// neighbour lane's register (warp=32 lanes, lane = linear % 32).
static std::vector<ExecState>* g_block_states = nullptr;

// Advance one thread by exactly one instruction. Returns:
//   1 = advanced, 2 = blocked at a block barrier (bar.*),
//   3 = finished (ret/exit), 0 = emulation error.
int run_thread_step(const PtxFunction& fn, const std::vector<uint8_t>& blob,
                    ExecState& st, std::string* error) {
  st.fn = &fn;
  st.params = blob.data();

  if (st.pc < 0 || st.pc >= static_cast<int>(fn.body.size())) {
    if (error) *error = "program counter out of range (missing ret?) in " + fn.name;
    return 0;
  }
  const PtxInstruction& ins = fn.body[st.pc];
  const int next_pc = st.pc + 1;

  // Predicate gate.
  if (!ins.pred.empty()) {
    std::string p = ins.pred;
    bool negate = false;
    if (p[0] == '!') { negate = true; p = p.substr(1); }
    const auto it = st.preds.find(p);
    const bool val = (it != st.preds.end()) && it->second;
    if (val == negate) { st.pc = next_pc; return 1; }
  }

  const std::string& op = ins.op;
  const auto& A = ins.args;

    if (op == "ret" || op == "exit") return 3;

    if (op == "bra" || op == "bra.uni") {
      if (A.empty()) { if (error) *error = "bra without target in " + fn.name; return 0; }
      const auto it = fn.labels.find(A[0]);
      if (it == fn.labels.end()) { if (error) *error = "unknown label '" + A[0] + "' in " + fn.name; return 0; }
      st.pc = it->second;
      return 1;
    }

    if (op.rfind("ld.param.", 0) == 0) {
      // ld.param.u64 %rd, [name]
      if (A.size() < 2) { if (error) *error = "bad ld.param in " + fn.name; return 0; }
      const PtxParam* p = find_param(fn, strip_brackets(A[1]));
      if (!p) { if (error) *error = "unknown param '" + A[1] + "' in " + fn.name; return 0; }
      st.setreg(A[0], read_mem(st.params + p->offset, p->width));
      st.pc = next_pc;
      return 1;
    }

    if (op.rfind("cvta.", 0) == 0) {
      // Host virtual addresses in emulation -> identity, except shared symbols
      // which resolve to the block scratchpad base + laid-out offset.
      if (A.size() >= 2) {
        uintptr_t shared = 0;
        if (shared_symbol_addr(fn, st, A[1], &shared))
          st.setreg(A[0], static_cast<uint64_t>(shared));
        else {
          uint64_t v = 0;
          if (!st.spec_reg(A[1], &v)) v = st.reg(A[1]);
          st.setreg(A[0], v);
        }
      }
      st.pc = next_pc;
      return 1;
    }

    if (op.rfind("mov.", 0) == 0) {
      if (A.size() < 2) { if (error) *error = "bad mov in " + fn.name; return 0; }
      st.setreg(A[0], operand_u64(st, A[1]));
      st.pc = next_pc;
      return 1;
    }

    if (op == "mul.wide.u32" || op == "mul.wide.s32") {
      if (A.size() < 3) { if (error) *error = "bad mul.wide in " + fn.name; return 0; }
      const uint32_t a = static_cast<uint32_t>(operand_u64(st, A[1]));
      const uint32_t b = static_cast<uint32_t>(operand_u64(st, A[2]));
      if (op == "mul.wide.s32") {
        // 32x32 -> 64 signed widening multiply (matches the AOT's sext+mul).
        const int64_t prod = static_cast<int64_t>(static_cast<int32_t>(a)) *
                             static_cast<int64_t>(static_cast<int32_t>(b));
        st.setreg(A[0], static_cast<uint64_t>(prod));
      } else {
        st.setreg(A[0], static_cast<uint64_t>(a) * static_cast<uint64_t>(b));
      }
      st.pc = next_pc;
      return 1;
    }

    if (op == "add.s64" || op == "add.u64" || op == "add.u32" || op == "add.s32") {
      if (A.size() < 3) { if (error) *error = "bad add in " + fn.name; return 0; }
      st.setreg(A[0], operand_u64(st, A[1]) + operand_u64(st, A[2]));
      st.pc = next_pc;
      return 1;
    }
    if (op == "sub.s64" || op == "sub.u64" || op == "sub.u32" || op == "sub.s32") {
      if (A.size() < 3) { if (error) *error = "bad sub in " + fn.name; return 0; }
      st.setreg(A[0], operand_u64(st, A[1]) - operand_u64(st, A[2]));
      st.pc = next_pc;
      return 1;
    }
    if (op == "add.f32" || op == "sub.f32" || op == "mul.f32" || op == "div.f32") {
      if (A.size() < 3) { if (error) *error = "bad fp op in " + fn.name; return 0; }
      const float x = bit_cast_v<float>(static_cast<uint32_t>(operand_u64(st, A[1])));
      const float y = bit_cast_v<float>(static_cast<uint32_t>(operand_u64(st, A[2])));
      float r = 0.0f;
      if (op == "add.f32") r = x + y;
      else if (op == "sub.f32") r = x - y;
      else if (op == "mul.f32") r = x * y;
      else r = x / y;
      st.setreg(A[0], static_cast<uint64_t>(bit_cast_v<uint32_t>(r)));
      st.pc = next_pc;
      return 1;
    }

    if (op == "add.f16" || op == "sub.f16" || op == "mul.f16" || op == "div.f16" ||
        op == "neg.f16" || op == "fma.rn.f16" ||
        op == "add.rn.bf16" || op == "sub.rn.bf16" || op == "mul.rn.bf16" ||
        op == "fma.rn.bf16" || op == "neg.bf16") {
      // f16/bf16 arithmetic. The interpreter evaluates in f32 and rounds once
      // to the storage format (RNE) — bit-identical to native half/bfloat
      // semantics for add/sub/mul/div and what LLVM emits for the AOT path.
      const bool bf = op.find("bf16") != std::string::npos;
      const auto load16 = [&](const std::string& tok) {
        const uint16_t bits = static_cast<uint16_t>(operand_u64(st, tok));
        return bf ? bf16_to_f32(bits) : f16_to_f32(bits);
      };
      const auto store16 = [&](const std::string& tok, float value) {
        const uint16_t bits = bf ? f32_to_bf16_rne(value) : f32_to_f16_rne(value);
        st.setreg(tok, bits);
      };
      const size_t need = op.rfind("fma", 0) == 0 ? 4 : (op.rfind("neg", 0) == 0 ? 2 : 3);
      if (A.size() < need) { if (error) *error = "bad " + op + " in " + fn.name; return 0; }
      const float x = load16(A[1]);
      if (need == 2) {
        store16(A[0], -x);
      } else {
        const float y = load16(A[2]);
        float r = 0.0f;
        if (need == 4) {
          const float z = load16(A[3]);
          r = std::fma(x, y, z);
        } else if (op[0] == 'a') r = x + y;
        else if (op[0] == 's') r = x - y;
        else if (op[0] == 'm') r = x * y;
        else r = x / y;  // div
        store16(A[0], r);
      }
      st.pc = next_pc;
      return 1;
    }

    // ---- integer bit logic: and / or / xor / not / shl / shr ---------------
    // and.pred / or.pred / not.pred end in a predicate type, so int_type_bits
    // returns 0 for them and they fall through to the predicate block below.
    if (op.rfind("and.", 0) == 0 || op.rfind("or.", 0) == 0 ||
        op.rfind("xor.", 0) == 0 || op.rfind("not.", 0) == 0 ||
        op.rfind("shl.", 0) == 0 || op.rfind("shr.", 0) == 0) {
      const int bits = int_type_bits(op);
      if (bits == 16 || bits == 32 || bits == 64) {
        const bool unary = op.rfind("not.", 0) == 0;
        const size_t need = unary ? 2 : 3;
        if (A.size() < need) {
          if (error) *error = "bad " + op + " in " + fn.name;
          return 0;
        }
        const uint64_t a = operand_u64(st, A[1]);
        const uint64_t b = unary ? 0 : operand_u64(st, A[2]);
        uint64_t r = a;
        if (op.rfind("and.", 0) == 0) r = a & b;
        else if (op.rfind("or.", 0) == 0) r = a | b;
        else if (op.rfind("xor.", 0) == 0) r = a ^ b;
        else if (unary) r = ~a;
        else {
          // PTX shift amounts are taken modulo the operand width.
          const unsigned sh =
              static_cast<unsigned>(b & (bits == 64 ? 63ull : 31ull));
          if (op.rfind("shl.", 0) == 0) r = a << sh;
          else if (is_signed_int_type(op))
            r = (bits == 64) ? sar64(a, sh) : sar32(static_cast<uint32_t>(a), sh);
          else
            r = (bits == 64) ? (a >> sh) : (static_cast<uint32_t>(a) >> sh);
        }
        st.setreg(A[0], width_mask(r, bits));
        st.pc = next_pc;
        return 1;
      }
    }

    // ---- integer multiply / multiply-add / divide / remainder ------------
    // mul.wide.* is handled above; f32/f16 forms by their own blocks.
    if (op.rfind("mul.lo.", 0) == 0 || op.rfind("mul.hi.", 0) == 0 ||
        op.rfind("mad.lo.", 0) == 0 || op.rfind("mad.hi.", 0) == 0 ||
        op.rfind("div.", 0) == 0 || op.rfind("rem.", 0) == 0) {
      const int bits = int_type_bits(op);
      if (bits == 32 || bits == 64) {
        const bool is_mad = op.rfind("mad.", 0) == 0;
        const size_t need = is_mad ? 4 : 3;
        if (A.size() < need) {
          if (error) *error = "bad " + op + " in " + fn.name;
          return 0;
        }
        const uint64_t a = operand_u64(st, A[1]);
        const uint64_t b = operand_u64(st, A[2]);
        const uint64_t c = is_mad ? operand_u64(st, A[3]) : 0;
        const bool sgn = is_signed_int_type(op);
        uint64_t r = 0;
        if (op.rfind("mul.", 0) == 0 || is_mad) {
          if (bits == 64) {
            // 64-bit product: the low word of the full 128-bit result, and
            // the addend joins it in 64-bit (modulo 2^64).
            r = a * b + (is_mad ? c : 0ull);
          } else {
            const uint64_t prod = static_cast<uint64_t>(static_cast<uint32_t>(a)) *
                                  static_cast<uint64_t>(static_cast<uint32_t>(b));
            if (op.find(".hi.") != std::string::npos) {
              // High word of the 64-bit product. nvcc only emits mad.hi with a
              // zero addend (it splits a 64-bit product into lo + hi), so the
              // high word alone is the result.
              r = sgn ? sar64(bit_cast_v<uint64_t>(
                                 static_cast<int64_t>(static_cast<int32_t>(static_cast<uint32_t>(a))) *
                                 static_cast<int64_t>(static_cast<int32_t>(static_cast<uint32_t>(b)))),
                             32)
                      : (prod >> 32);
            } else {
              r = prod + (is_mad ? c : 0ull);
            }
          }
        } else if (bits == 32) {
          // A zero divisor is undefined in PTX; return 0 instead of trapping.
          if (sgn) {
            const int32_t x = static_cast<int32_t>(static_cast<uint32_t>(a));
            const int32_t y = static_cast<int32_t>(static_cast<uint32_t>(b));
            if (y == 0) r = 0;
            else if (x == INT32_MIN && y == -1) r = static_cast<uint32_t>(x);  // wraps
            else if (op.rfind("rem.", 0) == 0) r = static_cast<uint32_t>(x % y);
            else r = static_cast<uint32_t>(x / y);
          } else {
            const uint32_t x = static_cast<uint32_t>(a);
            const uint32_t y = static_cast<uint32_t>(b);
            if (y == 0) r = 0;
            else if (op.rfind("rem.", 0) == 0) r = x % y;
            else r = x / y;
          }
        } else {
          if (sgn) {
            const int64_t x = static_cast<int64_t>(a);
            const int64_t y = static_cast<int64_t>(b);
            if (y == 0) r = 0;
            else if (x == INT64_MIN && y == -1) r = static_cast<uint64_t>(x);  // wraps
            else if (op.rfind("rem.", 0) == 0) r = static_cast<uint64_t>(x % y);
            else r = static_cast<uint64_t>(x / y);
          } else {
            if (b == 0) r = 0;
            else if (op.rfind("rem.", 0) == 0) r = a % b;
            else r = a / b;
          }
        }
        st.setreg(A[0], width_mask(r, bits));
        st.pc = next_pc;
        return 1;
      }
    }

    // ---- min / max --------------------------------------------------------
    if (op.rfind("min.", 0) == 0 || op.rfind("max.", 0) == 0) {
      const int bits = int_type_bits(op);
      if (bits == 32 || bits == 64) {
        if (A.size() < 3) {
          if (error) *error = "bad " + op + " in " + fn.name;
          return 0;
        }
        const uint64_t a = operand_u64(st, A[1]);
        const uint64_t b = operand_u64(st, A[2]);
        const bool is_min = op.rfind("min.", 0) == 0;
        uint64_t r = a;
        if (bits == 32) {
          if (is_signed_int_type(op)) {
            const int32_t x = static_cast<int32_t>(static_cast<uint32_t>(a));
            const int32_t y = static_cast<int32_t>(static_cast<uint32_t>(b));
            r = is_min ? (x < y ? x : y) : (x > y ? x : y);
          } else {
            const uint32_t x = static_cast<uint32_t>(a);
            const uint32_t y = static_cast<uint32_t>(b);
            r = is_min ? (x < y ? x : y) : (x > y ? x : y);
          }
        } else if (is_signed_int_type(op)) {
          const int64_t x = static_cast<int64_t>(a);
          const int64_t y = static_cast<int64_t>(b);
          r = static_cast<uint64_t>(is_min ? (x < y ? x : y) : (x > y ? x : y));
        } else {
          r = is_min ? (a < b ? a : b) : (a > b ? a : b);
        }
        st.setreg(A[0], width_mask(r, bits));
        st.pc = next_pc;
        return 1;
      }
    }

    // ---- bfi / bfe: bit-field insert / extract ----------------------------
    // nvcc lowers __shfl through bfi, so real PTX needs it even though
    // hand-written kernels rarely use it.
    //   bfi.b32 d, a, b, c : insert field b of a starting at bit c&0xff,
    //                       width (c>>8)&0xff  -> d
    //   bfe.b32 d, a, b, c : extract that field out of a
    if (op.rfind("bfi.", 0) == 0 || op.rfind("bfe.", 0) == 0) {
      if (A.size() < 4) { if (error) *error = "bad " + op + " in " + fn.name; return 0; }
      const uint64_t a = operand_u64(st, A[1]);
      const uint64_t b = operand_u64(st, A[2]);
      const uint32_t c = static_cast<uint32_t>(operand_u64(st, A[3]));
      const unsigned start = c & 0xffu;
      unsigned width = (c >> 8) & 0xffu;
      if (width == 0) width = 32;
      if (width > 32) width = 32;
      const uint32_t mask32 = (width >= 32) ? 0xffffffffu
                                            : ((1u << width) - 1u);
      uint64_t r;
      if (op.rfind("bfi.", 0) == 0) {
        const uint32_t av = static_cast<uint32_t>(a) & ~mask32;
        const uint32_t bv = static_cast<uint32_t>(b) & mask32;
        r = av | (static_cast<uint64_t>(bv) << start);
      } else {
        r = (static_cast<uint32_t>(a) >> start) & mask32;
      }
      int rb = 32;
      {
        const auto it = fn.regBits.find(A[0]);
        if (it != fn.regBits.end()) rb = it->second;
      }
      st.setreg(A[0], width_mask(r, rb));
      st.pc = next_pc;
      return 1;
    }

    // ---- neg / abs / popc / clz / bfind.shiftamt --------------------------
    if (op.rfind("neg.", 0) == 0 || op.rfind("abs.", 0) == 0 ||
        op.rfind("popc.", 0) == 0 || op.rfind("clz.", 0) == 0 ||
        op.rfind("bfind.shiftamt.", 0) == 0) {
      const int bits = int_type_bits(op);
      if (bits != 0) {
        if (A.size() < 2) {
          if (error) *error = "bad " + op + " in " + fn.name;
          return 0;
        }
        const uint64_t a = operand_u64(st, A[1]);
        uint64_t r = 0;
        if (op.rfind("neg.", 0) == 0) {
          r = 0ull - a;  // two's complement negation
        } else if (op.rfind("abs.", 0) == 0) {
          if (op.find(".u") != std::string::npos) {
            r = a & ~(1ull << (bits - 1));  // abs.u*: drop the sign bit
          } else if (bits == 32) {
            const int32_t x = static_cast<int32_t>(static_cast<uint32_t>(a));
            r = static_cast<uint32_t>(x < 0 ? 0u - static_cast<uint32_t>(x)
                                            : static_cast<uint32_t>(x));
          } else {
            // abs.s64: |x|, with INT64_MIN wrapping to itself.
            r = (a >> 63) ? (0ull - a) : a;
          }
        } else if (op.rfind("popc.", 0) == 0) {
          r = (bits == 64) ? popcount64(a) : popcount32(static_cast<uint32_t>(a));
        } else {
          // clz and bfind.shiftamt: distance from the top set bit (all-zero
          // input yields the full width, as on the hardware).
          r = (bits == 64) ? clz64(a) : clz32(static_cast<uint32_t>(a));
        }
        st.setreg(A[0], width_mask(r, bits));
        st.pc = next_pc;
        return 1;
      }
    }

    // ---- f32 unary: fneg / fnabs / sqrt / rsqrt / rcp ---------------------
    if (op == "fneg.f32" || op == "fnabs.f32" ||
        ((op.rfind("sqrt.", 0) == 0 || op.rfind("rsqrt.", 0) == 0 ||
          op.rfind("rcp.", 0) == 0) &&
         op.find(".f32") != std::string::npos)) {
      if (A.size() < 2) {
        if (error) *error = "bad " + op + " in " + fn.name;
        return 0;
      }
      const uint32_t in = static_cast<uint32_t>(operand_u64(st, A[1]));
      uint32_t r = in;
      if (op == "fneg.f32") {
        r = in ^ 0x80000000u;  // sign flip (exact for +-0, inf and NaN too)
      } else if (op == "fnabs.f32") {
        r = in & 0x7fffffffu;
      } else {
        const float x = bit_cast_v<float>(in);
        float y = 0.0f;
        if (op.rfind("rsqrt.", 0) == 0) y = 1.0f / std::sqrt(x);
        else if (op.rfind("rcp.", 0) == 0) y = 1.0f / x;
        else y = std::sqrt(x);
        r = bit_cast_v<uint32_t>(y);
      }
      st.setreg(A[0], r);
      st.pc = next_pc;
      return 1;
    }

    // ---- selp / slct: per-thread select -----------------------------------
    if (op.rfind("selp.", 0) == 0) {
      // selp.<type> %d, %a, %b, %p  ->  %p ? %a : %b
      if (A.size() < 4) {
        if (error) *error = "bad " + op + " in " + fn.name;
        return 0;
      }
      const auto it = st.preds.find(A[3]);
      const bool take_a = (it != st.preds.end()) && it->second;
      const int bits = int_type_bits(op);
      st.setreg(A[0], width_mask(operand_u64(st, take_a ? A[1] : A[2]), bits ? bits : 64));
      st.pc = next_pc;
      return 1;
    }

    if (op.rfind("slct.", 0) == 0) {
      // slct.<type> %d, %a, %b, %c  ->  (c & 1) ? %a : %b
      if (A.size() < 4) {
        if (error) *error = "bad " + op + " in " + fn.name;
        return 0;
      }
      const bool take_a = (operand_u64(st, A[3]) & 1ull) != 0;
      const int bits = int_type_bits(op);
      st.setreg(A[0], width_mask(operand_u64(st, take_a ? A[1] : A[2]), bits ? bits : 64));
      st.pc = next_pc;
      return 1;
    }

    // ---- predicate logic: and.pred / or.pred / not.pred -------------------
    if (op.rfind("and.pred", 0) == 0 || op.rfind("or.pred", 0) == 0) {
      // and.pred d, a, b[, c]  ->  (a && b) || !c     (c defaults to true)
      // or.pred  d, a, b[, c]  ->  (a || b) || !c
      if (A.size() < 3) {
        if (error) *error = "bad " + op + " in " + fn.name;
        return 0;
      }
      const auto value = [&](const std::string& n) {
        const auto it = st.preds.find(n);
        return it != st.preds.end() && it->second;
      };
      const bool a = value(A[1]);
      const bool b = value(A[2]);
      const bool c = A.size() >= 4 ? value(A[3]) : true;
      st.preds[A[0]] = (op[0] == 'a') ? ((a && b) || !c) : ((a || b) || !c);
      st.pc = next_pc;
      return 1;
    }

    if (op.rfind("not.pred", 0) == 0 || op.rfind("cnot.pred", 0) == 0) {
      // not.pred d, a[, c]  ->  (a != c); with the 2-operand form c is true,
      // which reduces to !a.
      if (A.size() < 2) {
        if (error) *error = "bad " + op + " in " + fn.name;
        return 0;
      }
      const auto value = [&](const std::string& n) {
        const auto it = st.preds.find(n);
        return it != st.preds.end() && it->second;
      };
      const bool a = value(A[1]);
      const bool c = A.size() >= 3 ? value(A[2]) : true;
      st.preds[A[0]] = (a != c);
      st.pc = next_pc;
      return 1;
    }

    if (op.find("x2") != std::string::npos &&
        op.rfind("ld.", 0) != 0 && op.rfind("st.", 0) != 0) {
      // f16x2/bf16x2 fallback for non-memory ops: payload as i32
      if (A.size() >= 2) {
        if (op.rfind("cvt.", 0) == 0 || op.rfind("mov.", 0) == 0) {
          st.setreg(A[0], operand_u64(st, A[1]));
        } else if (op.rfind("add.", 0) == 0 || op.rfind("sub.", 0) == 0 ||
                   op.rfind("mul.", 0) == 0 || op.rfind("fma.", 0) == 0) {
          if (A.size() >= 3 && op.rfind("add.", 0) == 0)
            st.setreg(A[0], operand_u64(st, A[1]) + operand_u64(st, A[2]));
          else
            st.setreg(A[0], operand_u64(st, A[1]));
        } else {
          st.setreg(A[0], operand_u64(st, A[1]));
        }
        st.pc = next_pc;
        return 1;
      }
    }

    if (op.rfind("cvt.", 0) == 0) {
      // cvt.<mod>.<dst>.<src> %d, %s  (integer conversions; a few float cases
      // for realism)
      if (A.size() < 2) { if (error) *error = "bad cvt in " + fn.name; return 0; }
      const std::string full = op.substr(4);  // "<mod>.<dst>.<src>"

      // Saturating and explicitly-rounded conversions (cvt.sat.*, cvt.rz/rn/
      // ru/rp/rm/rzi.*) are deliberately not emulated: they need PTX's
      // saturation and rounding rules, so fail loudly instead of silently
      // handing back the source bits. ".rn" stays legal for the f16/bf16
      // narrowing handled below.
      {
        std::stringstream mods(full);
        std::string mt;
        while (std::getline(mods, mt, '.')) {
          const bool is_round = mt == "rn" || mt == "rz" || mt == "rm" ||
                                mt == "rp" || mt == "ru" || mt == "rzi";
          if (mt != "sat" && !is_round) continue;
          if (mt == "rn" && (op.find(".f16") != std::string::npos ||
                             op.find(".bf16") != std::string::npos))
            continue;
          if (error)
            *error = "unsupported rounding/saturating conversion '" + op +
                     "' in " + fn.name;
          return 0;
        }
      }

      std::string spec = full;
      if (spec.rfind("rn.", 0) == 0) spec = spec.substr(3);  // cvt.rn.f16.f32 etc.
      const size_t dot = spec.find('.');
      const std::string dst = dot == std::string::npos ? spec : spec.substr(0, dot);
      const std::string src = dot == std::string::npos ? "" : spec.substr(dot + 1);
      const uint64_t raw = operand_u64(st, A[1]);
      const uint32_t raw32 = static_cast<uint32_t>(raw);

      // f16 / bf16 conversions (quantization is where the AOT path needs them
      // for FP16/BF16 GEMM tiles; the RNE narrowing matches LLVM fptrunc).
      if (src == "f16" && (dst == "f32" || dst == "f64")) {
        st.setreg(A[0], bit_cast_v<uint32_t>(f16_to_f32(static_cast<uint16_t>(raw))));
        st.pc = next_pc;
        return 1;
      }
      if (src == "bf16" && (dst == "f32" || dst == "f64")) {
        st.setreg(A[0], bit_cast_v<uint32_t>(bf16_to_f32(static_cast<uint16_t>(raw))));
        st.pc = next_pc;
        return 1;
      }
      if (dst == "f16" && (src == "f32" || src == "f64")) {
        st.setreg(A[0], f32_to_f16_rne(bit_cast_v<float>(raw32)));
        st.pc = next_pc;
        return 1;
      }
      if (dst == "bf16" && (src == "f32" || src == "f64")) {
        st.setreg(A[0], f32_to_bf16_rne(bit_cast_v<float>(raw32)));
        st.pc = next_pc;
        return 1;
      }
      if (dst == "f32" && (src == "u32" || src == "s32" || src == "b32" ||
                           src == "u16" || src == "s16" || src == "b16")) {
        // 32-bit source: signed/unsigned as written; 16-bit source: widened to
        // 32 first (sign-extended for .s16) so the float value is exact.
        float fv = 0.0f;
        if (src == "s32") fv = static_cast<float>(static_cast<int32_t>(raw32));
        else if (src == "u32" || src == "b32") fv = static_cast<float>(raw32);
        else if (src == "s16")
          fv = static_cast<float>(static_cast<int16_t>(static_cast<uint16_t>(raw)));
        else fv = static_cast<float>(static_cast<uint16_t>(raw));
        st.setreg(A[0], bit_cast_v<uint32_t>(fv));
        st.pc = next_pc;
        return 1;
      }
      if (dst == "f32" && (src == "u64" || src == "s64" || src == "b64")) {
        const float fv = (src == "u64")
                             ? static_cast<float>(raw)
                             : static_cast<float>(static_cast<int64_t>(raw));
        st.setreg(A[0], bit_cast_v<uint32_t>(fv));
        st.pc = next_pc;
        return 1;
      }
      if (dst == "f64" && (src == "u32" || src == "s32" || src == "b32" ||
                           src == "u16" || src == "s16" || src == "b16" ||
                           src == "u64" || src == "s64" || src == "b64")) {
        double dv = 0.0;
        if (src == "s16")
          dv = static_cast<double>(static_cast<int16_t>(static_cast<uint16_t>(raw)));
        else if (src == "s32") dv = static_cast<double>(static_cast<int32_t>(raw32));
        else if (src == "s64") dv = static_cast<double>(static_cast<int64_t>(raw));
        else dv = static_cast<double>(raw);
        st.setreg(A[0], bit_cast_v<uint64_t>(dv));
        st.pc = next_pc;
        return 1;
      }
      if (src == "f64" && (dst == "u32" || dst == "s32" || dst == "b32" ||
                           dst == "u16" || dst == "s16" || dst == "b16" ||
                           dst == "u64" || dst == "s64" || dst == "b64")) {
        // float -> integer: round toward zero and saturate at the destination
        // range (NaN -> 0), like the hardware fptosi/fptoui.
        const int bits = int_type_bits("cvt." + dst);
        st.setreg(A[0], width_mask(fp_to_int(bit_cast_v<double>(raw), bits,
                                             dst[0] == 's'),
                                   bits));
        st.pc = next_pc;
        return 1;
      }
      if (src == "f32" && (dst == "u64" || dst == "s64" || dst == "b64")) {
        st.setreg(A[0], fp_to_int(bit_cast_v<float>(raw32), 64, dst[0] == 's'));
        st.pc = next_pc;
        return 1;
      }
      if (dst == "f32" && src == "f64") {
        st.setreg(A[0], bit_cast_v<uint32_t>(static_cast<float>(bit_cast_v<double>(raw))));
        st.pc = next_pc;
        return 1;
      }
      if (dst == "f64" && src == "f32") {
        st.setreg(A[0], bit_cast_v<uint64_t>(static_cast<double>(bit_cast_v<float>(raw32))));
        st.pc = next_pc;
        return 1;
      }
      // 16-bit destinations keep the low halfword whatever the source width.
      if ((dst == "u16" || dst == "s16" || dst == "b16") &&
          (src == "u16" || src == "s16" || src == "b16" ||
           src == "u32" || src == "s32" || src == "b32" ||
           src == "u64" || src == "s64" || src == "b64")) {
        st.setreg(A[0], raw & 0xffffu);
        st.pc = next_pc;
        return 1;
      }
      // 16 -> 32: zero-extend for .u16/.b16, sign-extend for .s16.
      if ((dst == "u32" || dst == "s32" || dst == "b32") &&
          (src == "u16" || src == "s16" || src == "b16")) {
        st.setreg(A[0], width_mask(src == "s16" ? sext64(raw, 16) : raw, 32));
        st.pc = next_pc;
        return 1;
      }
      // 64 -> 32: PTX narrows to the destination width, so .s64 keeps the sign
      // in bit 31 of the result.
      if ((dst == "u32" || dst == "s32") && src == "s64") {
        st.setreg(A[0], static_cast<uint32_t>(sext64(raw, 32)));
        st.pc = next_pc;
        return 1;
      }
      // 16 -> 64: zero-extend for .u16, sign-extend for .s16.
      if ((dst == "u64" || dst == "s64" || dst == "b64") &&
          (src == "u16" || src == "s16" || src == "b16")) {
        st.setreg(A[0], src == "s16" ? sext64(raw, 16) : (raw & 0xffffu));
        st.pc = next_pc;
        return 1;
      }
      if (src == "f16" && (dst == "u32" || dst == "s32" || dst == "u16" || dst == "s16")) {
        const float f = f16_to_f32(static_cast<uint16_t>(raw));
        const uint32_t iv = static_cast<uint32_t>(static_cast<int32_t>(f));
        st.setreg(A[0], iv);
        st.pc = next_pc;
        return 1;
      }
      if (src == "bf16" && (dst == "u32" || dst == "s32" || dst == "u16" || dst == "s16")) {
        const float f = bf16_to_f32(static_cast<uint16_t>(raw));
        const uint32_t iv = static_cast<uint32_t>(static_cast<int32_t>(f));
        st.setreg(A[0], iv);
        st.pc = next_pc;
        return 1;
      }
      if ((dst == "f16" || dst == "bf16") &&
          (src == "u32" || src == "s32" || src == "u16" || src == "s16")) {
        const float f = (src[0] == 'u') ? static_cast<float>(raw32)
                                        : static_cast<float>(static_cast<int32_t>(raw));
        st.setreg(A[0], dst == "f16" ? f32_to_f16_rne(f) : f32_to_bf16_rne(f));
        st.pc = next_pc;
        return 1;
      }
      if (dst == "f16" && src == "bf16") {
        st.setreg(A[0], f16_to_f32(static_cast<uint16_t>(f32_to_bf16_rne(
                       bf16_to_f32(static_cast<uint16_t>(raw))))));
        st.pc = next_pc;
        return 1;
      }
      if (dst == "bf16" && src == "f16") {
        st.setreg(A[0], f32_to_bf16_rne(f16_to_f32(static_cast<uint16_t>(raw))));
        st.pc = next_pc;
        return 1;
      }

      const bool src64 = (src == "u64" || src == "s64" || src == "b64" || src == "f64");
      uint64_t v = 0;
      if (src64) {
        if (dst == "u32" || dst == "s32" || dst == "b32") v = raw32;  // truncate
        else if (dst == "f32") v = bit_cast_v<uint32_t>(static_cast<float>(static_cast<int64_t>(raw)));
        else v = raw;
      } else {
        if (dst == "u64" || dst == "s64" || dst == "b64") v = raw32;  // zero-extend
        else if (dst == "f32") v = bit_cast_v<uint32_t>(static_cast<float>(raw32));
        else v = raw32;
      }
      st.setreg(A[0], v);
      st.pc = next_pc;
      return 1;
    }

    if (op.rfind("setp.", 0) == 0) {
      if (A.size() < 3) { if (error) *error = "bad setp in " + fn.name; return 0; }
      const uint32_t x = static_cast<uint32_t>(operand_u64(st, A[1]));
      const uint32_t y = static_cast<uint32_t>(operand_u64(st, A[2]));
      const size_t d1 = op.find('.');
      std::string cond = (d1 == std::string::npos) ? "ge" : op.substr(d1 + 1);
      const size_t d2 = cond.find('.');
      if (d2 != std::string::npos) cond = cond.substr(0, d2);
      bool r = false;
      if (cond == "ge") r = x >= y;
      else if (cond == "gt") r = x > y;
      else if (cond == "lt") r = x < y;
      else if (cond == "le") r = x <= y;
      else if (cond == "eq") r = x == y;
      else if (cond == "ne") r = x != y;
      else {
        if (error) *error = "unsupported setp condition '" + cond + "' in " + fn.name;
        return false;
      }
      st.preds[A[0]] = r;
      st.pc = next_pc;
      return 1;
    }

    if (op.rfind("ld.global.", 0) == 0 || op.rfind("ld.shared.", 0) == 0) {
      // ld.global.f32 %r, [%rd] (also ld.global.nc.*); ld.shared.f32 %r, [%rd]
      if (A.size() < 2) { if (error) *error = "bad " + op + " in " + fn.name; return 0; }
      const uintptr_t addr = mem_addr(fn, st, A[1]);
      std::string pref;
      if (op.rfind("ld.global.nc.", 0) == 0) pref = "ld.global.nc.";
      else if (op.rfind("ld.global.", 0) == 0) pref = "ld.global.";
      else pref = "ld.shared.";
      const std::string ty = op.substr(pref.size());
      if (ty == "v2.u32" || ty == "v2.s32" || ty == "v2.b32" || ty == "v2.f32") {
        // 64-bit vector load. Operands are split on ',', so a destination
        // list "{%r2, %r3}" arrives as A[0]="{%r2", A[1]="r3}" and the
        // address moves to A[2]; a plain "%rd" destination keeps two operands.
        const uintptr_t vaddr = mem_addr(fn, st, A.size() >= 3 ? A[2] : A[1]);
        const uint64_t v64 = read_mem(reinterpret_cast<const uint8_t*>(vaddr), 8);
        if (A.size() >= 3) {
          st.setreg(vector_reg_name(A[0]), v64 & 0xffffffffu);
          st.setreg(vector_reg_name(A[1]), static_cast<uint32_t>(v64 >> 32));
        } else {
          st.setreg(A[0], v64);
        }
      } else if (ty == "u8" || ty == "s8" || ty == "b8") {
        // 1-byte load; .s8 sign-extends into the destination register.
        const uint8_t b8 = *reinterpret_cast<const uint8_t*>(addr);
        st.setreg(A[0], (ty == "s8")
                           ? static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(b8)))
                           : static_cast<uint32_t>(b8));
      } else if (ty == "f32" || ty == "b32" || ty == "u32" || ty == "s32") {
        st.setreg(A[0], read_mem(reinterpret_cast<const uint8_t*>(addr), 4) & 0xffffffffu);
      } else if (ty == "u64" || ty == "b64" || ty == "s64") {
        st.setreg(A[0], read_mem(reinterpret_cast<const uint8_t*>(addr), 8));
      } else if (ty == "s16") {
        // 16-bit signed load: sign-extend (PTX .s16 carries the sign).
        const uint16_t h = static_cast<uint16_t>(read_mem(reinterpret_cast<const uint8_t*>(addr), 2));
        st.setreg(A[0], static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(h))));
      } else if (ty == "f16" || ty == "b16" || ty == "u16" ||
                 ty == "bf16" || ty == "f16x2" || ty == "bf16x2") {
        const int bytes = (ty == "f16x2" || ty == "bf16x2") ? 4 : 2;
        st.setreg(A[0], read_mem(reinterpret_cast<const uint8_t*>(addr), bytes) & 0xffffffffu);
      } else {
        if (error) *error = "unsupported " + pref + " type '" + ty + "' in " + fn.name;
        return 0;
      }
      st.pc = next_pc;
      return 1;
    }

    if (op.rfind("st.global.", 0) == 0 || op.rfind("st.shared.", 0) == 0) {
      // st.global.f32 [%rd], %r ; st.shared.f32 [%rd], %r
      if (A.size() < 2) { if (error) *error = "bad " + op + " in " + fn.name; return 0; }
      const uintptr_t addr = mem_addr(fn, st, A[0]);
      const std::string pref =
          op.rfind("st.global.", 0) == 0 ? "st.global." : "st.shared.";
      const std::string ty = op.substr(pref.size());
      const uint64_t v = st.reg(A[1]);
      auto* p = reinterpret_cast<uint8_t*>(addr);
      if (ty == "v2.u32" || ty == "v2.s32" || ty == "v2.b32" || ty == "v2.f32") {
        // 64-bit vector store: the source list "{%r2, %r3}" is split into
        // A[1]="{%r2", A[2]="r3}" (low word first, as in memory).
        uint64_t v64 = v;
        if (A.size() >= 3) {
          v64 = static_cast<uint64_t>(static_cast<uint32_t>(st.reg(vector_reg_name(A[1])))) |
                (static_cast<uint64_t>(static_cast<uint32_t>(st.reg(vector_reg_name(A[2])))) << 32);
        }
        std::memcpy(p, &v64, 8);
      } else if (ty == "u8" || ty == "s8" || ty == "b8") {
        const uint8_t b8 = static_cast<uint8_t>(v);
        std::memcpy(p, &b8, 1);
      } else if (ty == "f32") {
        const uint32_t bits = static_cast<uint32_t>(v);
        std::memcpy(p, &bits, 4);
      } else if (ty == "u32" || ty == "b32" || ty == "s32") {
        const uint32_t bits = static_cast<uint32_t>(v);
        std::memcpy(p, &bits, 4);
      } else if (ty == "u64" || ty == "b64" || ty == "s64") {
        std::memcpy(p, &v, 8);
      } else if (ty == "f16" || ty == "b16" || ty == "u16" || ty == "s16" ||
                 ty == "bf16" || ty == "f16x2" || ty == "bf16x2") {
        const int bytes = (ty == "f16x2" || ty == "bf16x2") ? 4 : 2;
        const uint64_t val = bytes == 4 ? (v & 0xffffffffu) : (v & 0xffffu);
        std::memcpy(p, &val, static_cast<size_t>(bytes));
      } else {
        if (error) *error = "unsupported " + pref + " type '" + ty + "' in " + fn.name;
        return 0;
      }
      st.pc = next_pc;
      return 1;
    }

    if (op.rfind("atom.", 0) == 0 || op.rfind("red.", 0) == 0) {
      // Supported forms (scope/space qualifiers are accepted and ignored):
      //   atom.add.{u32,s32,u64,s64,f32} %r, [addr], %val  -> old
      //   red.add.{...} [addr], %val                        -> no result
      //   atom.cas.b32 %r, [addr], %cmp, %new               -> old (cmpxchg)
      //   atom.cas.b64 %r, [addr], %cmp, %new
      //   atom.exch.b32 %r, [addr], %val                    -> old (swap)
      //   atom.min/max.{u32,s32,u64,s64,f32} %r, [addr], %val -> old
      //   red.min/max.{...} [addr], %val                    -> no result
      const bool has_dst = op.rfind("atom.", 0) == 0;
      const bool is_cas = op.find(".cas.") != std::string::npos;
      const bool is_exch = op.find(".exch.") != std::string::npos;
      const bool is_min = op.find(".min.") != std::string::npos;
      const bool is_max = op.find(".max.") != std::string::npos;
      const bool is_add = op.find(".add.") != std::string::npos;
      if (!is_cas && !is_exch && !is_min && !is_max && !is_add) {
        if (error) *error = "unsupported atomic '" + op + "' in " + fn.name;
        return 0;
      }
      const std::string ty = op.substr(op.rfind('.') + 1);
      if (is_cas) {
        const size_t ai = 1; // cas is always atom (has dst)
        if (!has_dst) { if (error) *error = "red.cas not supported '" + op + "' in " + fn.name; return 0; }
        if (A.size() < 4) { if (error) *error = "bad " + op + " in " + fn.name; return 0; }
        const uintptr_t addr = mem_addr(fn, st, A[ai]);
        auto* p = reinterpret_cast<uint8_t*>(addr);
        if (ty == "b32" || ty == "u32" || ty == "s32" || ty == "f32") {
          uint32_t cur = 0; std::memcpy(&cur, p, 4);
          const uint32_t cmp = static_cast<uint32_t>(operand_u64(st, A[ai + 1]));
          const uint32_t nw = static_cast<uint32_t>(operand_u64(st, A[ai + 2]));
          const uint32_t old = cur;
          if (cur == cmp) std::memcpy(p, &nw, 4);
          st.setreg(A[0], old);
        } else if (ty == "b64" || ty == "u64" || ty == "s64" || ty == "f64") {
          uint64_t cur = 0; std::memcpy(&cur, p, 8);
          const uint64_t cmp = operand_u64(st, A[ai + 1]);
          const uint64_t nw = operand_u64(st, A[ai + 2]);
          const uint64_t old = cur;
          if (cur == cmp) std::memcpy(p, &nw, 8);
          st.setreg(A[0], old);
        } else {
          if (error) *error = "unsupported atomic type '" + ty + "' in " + fn.name;
          return 0;
        }
        st.pc = next_pc;
        return 1;
      }
      if (is_exch) {
        if (!has_dst) { if (error) *error = "red.exch not supported '" + op + "' in " + fn.name; return 0; }
        const size_t ai = 1;
        if (A.size() < 3) { if (error) *error = "bad " + op + " in " + fn.name; return 0; }
        const uintptr_t addr = mem_addr(fn, st, A[ai]);
        auto* p = reinterpret_cast<uint8_t*>(addr);
        const uint64_t val = operand_u64(st, A[ai + 1]);
        if (ty == "b32" || ty == "u32" || ty == "s32" || ty == "f32") {
          uint32_t cur = 0; std::memcpy(&cur, p, 4);
          const uint32_t nv = static_cast<uint32_t>(val);
          std::memcpy(p, &nv, 4);
          st.setreg(A[0], cur);
        } else if (ty == "b64" || ty == "u64" || ty == "s64" || ty == "f64") {
          uint64_t cur = 0; std::memcpy(&cur, p, 8);
          std::memcpy(p, &val, 8);
          st.setreg(A[0], cur);
        } else {
          if (error) *error = "unsupported atomic type '" + ty + "' in " + fn.name;
          return 0;
        }
        st.pc = next_pc;
        return 1;
      }
      if (is_min || is_max) {
        const size_t ai = has_dst ? 1 : 0;
        if (A.size() < ai + 2) { if (error) *error = "bad " + op + " in " + fn.name; return 0; }
        const uintptr_t addr = mem_addr(fn, st, A[ai]);
        auto* p = reinterpret_cast<uint8_t*>(addr);
        const uint64_t val = operand_u64(st, A[ai + 1]);
        uint64_t old = 0;
        const bool do_min = is_min;
        if (ty == "u32" || ty == "b32") {
          uint32_t cur = 0; std::memcpy(&cur, p, 4);
          const uint32_t v = static_cast<uint32_t>(val);
          const uint32_t res = do_min ? (cur < v ? cur : v) : (cur > v ? cur : v);
          std::memcpy(p, &res, 4);
          old = cur;
        } else if (ty == "s32") {
          int32_t cur = 0; std::memcpy(&cur, p, 4);
          const int32_t v = static_cast<int32_t>(static_cast<uint32_t>(val));
          const int32_t res = do_min ? (cur < v ? cur : v) : (cur > v ? cur : v);
          std::memcpy(p, &res, 4);
          old = static_cast<uint32_t>(cur);
        } else if (ty == "u64" || ty == "b64") {
          uint64_t cur = 0; std::memcpy(&cur, p, 8);
          const uint64_t res = do_min ? (cur < val ? cur : val) : (cur > val ? cur : val);
          std::memcpy(p, &res, 8);
          old = cur;
        } else if (ty == "s64") {
          int64_t cur = 0; std::memcpy(&cur, p, 8);
          const int64_t v = static_cast<int64_t>(val);
          const int64_t res = do_min ? (cur < v ? cur : v) : (cur > v ? cur : v);
          std::memcpy(p, &res, 8);
          old = static_cast<uint64_t>(cur);
        } else if (ty == "f32") {
          float cur = 0; std::memcpy(&cur, p, 4);
          const float v = bit_cast_v<float>(static_cast<uint32_t>(val));
          const float res = do_min ? fminf(cur, v) : fmaxf(cur, v);
          std::memcpy(p, &res, 4);
          old = bit_cast_v<uint32_t>(cur);
        } else if (ty == "f64") {
          double cur = 0; std::memcpy(&cur, p, 8);
          const double v = bit_cast_v<double>(val);
          const double res = do_min ? fmin(cur, v) : fmax(cur, v);
          std::memcpy(p, &res, 8);
          old = bit_cast_v<uint64_t>(cur);
        } else {
          if (error) *error = "unsupported atomic type '" + ty + "' in " + fn.name;
          return 0;
        }
        if (has_dst) st.setreg(A[0], old);
        st.pc = next_pc;
        return 1;
      }
      // is_add
      const size_t ai = has_dst ? 1 : 0;
      if (A.size() < ai + 2) { if (error) *error = "bad " + op + " in " + fn.name; return 0; }
      const uintptr_t addr = mem_addr(fn, st, A[ai]);
      const uint64_t val = operand_u64(st, A[ai + 1]);
      auto* p = reinterpret_cast<uint8_t*>(addr);
      uint64_t old = 0;
      if (ty == "u32" || ty == "s32" || ty == "b32") {
        uint32_t cur = 0;
        std::memcpy(&cur, p, 4);
        const uint32_t sum = cur + static_cast<uint32_t>(val);
        std::memcpy(p, &sum, 4);
        old = cur;
      } else if (ty == "u64" || ty == "s64" || ty == "b64") {
        uint64_t cur = 0;
        std::memcpy(&cur, p, 8);
        const uint64_t sum = cur + val;
        std::memcpy(p, &sum, 8);
        old = cur;
      } else if (ty == "f32") {
        float cur = 0.0f;
        std::memcpy(&cur, p, 4);
        const float sum = cur + bit_cast_v<float>(static_cast<uint32_t>(val));
        std::memcpy(p, &sum, 4);
        old = bit_cast_v<uint32_t>(cur);
      } else {
        if (error) *error = "unsupported atomic type '" + ty + "' in " + fn.name;
        return 0;
      }
      if (has_dst) st.setreg(A[0], old);
      st.pc = next_pc;
      return 1;
    }

    // ---- warp shuffles: shfl.sync.{up,down,bfly,idx}.b32 (and without .sync) ----
    // PTX: shfl.sync.<kind>.b32 d, a, b, c, mask  (or 4 args without mask)
    // Simplified: warp=32 lanes, lane = linear % 32, warpBase = linear - lane.
    // idx: target = b & 0x1f, bfly: lane ^ offset, up: lane - offset, down: lane + offset.
    // Out-of-range or inactive lanes keep their own source value.
    if (op.rfind("shfl", 0) == 0) {
      if (A.size() < 3) {
        if (error) *error = "bad " + op + " in " + fn.name;
        return 0;
      }
      // The destination may be predicated: "shfl.sync.down.b32 %r10|%p1, ..."
      // writes %r10 only when %p1 holds. Split it and honour the guard.
      std::string dst = A[0];
      std::string writePred;
      {
        const size_t bar = dst.find('|');
        if (bar != std::string::npos) {
          writePred = dst.substr(bar + 1);
          dst = dst.substr(0, bar);
          if (!writePred.empty() && writePred[0] == '%') writePred = writePred.substr(1);
        }
      }
      const std::string src = A[1];
      const bool is_idx = op.find(".idx.") != std::string::npos;
      const bool is_bfly = op.find(".bfly.") != std::string::npos;
      const bool is_up = op.find(".up.") != std::string::npos;
      const bool is_down = op.find(".down.") != std::string::npos;
      const uint32_t sel = static_cast<uint32_t>(operand_u64(st, A[2]));
      const uint32_t linear = st.tid[0] + st.tid[1] * st.ntid[0] + st.tid[2] * st.ntid[0] * st.ntid[1];
      const uint32_t lane = linear % 32;
      const uint32_t warpBase = linear - lane;
      uint32_t targetLane = 0;
      bool oob = false;
      if (is_idx) targetLane = sel & 0x1f;
      else if (is_bfly) targetLane = lane ^ (sel & 0x1f);
      else if (is_up) {
        const uint32_t off = sel & 0x1f;
        if (lane < off) oob = true;
        else targetLane = lane - off;
      } else if (is_down) {
        const uint32_t off = sel & 0x1f;
        targetLane = lane + off;
        if (targetLane >= 32) oob = true;
      } else {
        targetLane = sel & 0x1f;
      }
      uint64_t val = st.reg(src);
      if (!oob && g_block_states) {
        const size_t targetLinear = static_cast<size_t>(warpBase + targetLane);
        if (targetLinear < g_block_states->size()) {
          // For partial warps (block size not multiple of 32) the target lane
          // may be beyond the block; treat as out-of-range -> keep own value.
          const ExecState& srcState = (*g_block_states)[targetLinear];
          auto it = srcState.regs.find(src);
          if (it != srcState.regs.end()) val = it->second;
          else {
            // src may be an immediate? already handled via operand_u64 above, but
            // for register source we copy neighbour's register.
            uint64_t v = 0;
            if (srcState.spec_reg(src, &v)) val = v;
          }
        } else {
          oob = true;
        }
      }
      if (oob) val = st.reg(src);
      // Predicated write: leave the destination untouched when the guard is false.
      if (writePred.empty() || st.preds[writePred]) st.setreg(dst, val);
      st.pc = next_pc;
      return 1;
    }

    // ---- vote.{any,all,uni}.pred and vote.sync.* ----
    if (op.rfind("vote", 0) == 0) {
      if (A.size() < 2) {
        if (error) *error = "bad " + op + " in " + fn.name;
        return 0;
      }
      const std::string dst = A[0];
      std::string srcPred = A[1];
      bool srcNeg = false;
      if (!srcPred.empty() && srcPred[0] == '!') {
        srcNeg = true;
        srcPred = srcPred.substr(1);
      }
      const bool is_any = op.find(".any.") != std::string::npos;
      const bool is_all = op.find(".all.") != std::string::npos;
      const bool is_uni = op.find(".uni.") != std::string::npos;
      const uint32_t linear = st.tid[0] + st.tid[1] * st.ntid[0] + st.tid[2] * st.ntid[0] * st.ntid[1];
      const uint32_t lane = linear % 32;
      const uint32_t warpBase = linear - lane;
      bool any = false;
      bool all = true;
      bool uni = true;
      bool firstVal = false;
      bool haveFirst = false;
      int activeCount = 0;
      if (g_block_states) {
        for (uint32_t l = 0; l < 32; ++l) {
          const size_t idx = static_cast<size_t>(warpBase + l);
          if (idx >= g_block_states->size()) continue;
          const ExecState& s = (*g_block_states)[idx];
          auto it = s.preds.find(srcPred);
          bool v = (it != s.preds.end()) && it->second;
          if (srcNeg) v = !v;
          if (!haveFirst) {
            firstVal = v;
            haveFirst = true;
          } else if (v != firstVal) {
            uni = false;
          }
          any = any || v;
          all = all && v;
          ++activeCount;
        }
        if (activeCount == 0) {
          any = false;
          all = false;
          uni = true;
        }
      } else {
        auto it = st.preds.find(srcPred);
        bool v = (it != st.preds.end()) && it->second;
        if (srcNeg) v = !v;
        any = v;
        all = v;
        uni = true;
      }
      bool result = false;
      if (is_any) result = any;
      else if (is_all) result = all;
      else if (is_uni) result = uni;
      else result = any;
      st.preds[dst] = result;
      st.pc = next_pc;
      return 1;
    }

    // ---- activemask.b32 ----
    if (op.rfind("activemask", 0) == 0) {
      if (A.empty()) {
        if (error) *error = "bad " + op + " in " + fn.name;
        return 0;
      }
      const std::string dst = A[0];
      const uint32_t linear = st.tid[0] + st.tid[1] * st.ntid[0] + st.tid[2] * st.ntid[0] * st.ntid[1];
      const uint32_t lane = linear % 32;
      const uint32_t warpBase = linear - lane;
      uint32_t mask = 0;
      if (g_block_states) {
        for (uint32_t l = 0; l < 32; ++l) {
          const size_t idx = static_cast<size_t>(warpBase + l);
          if (idx < g_block_states->size()) mask |= (1u << l);
        }
      } else {
        mask = 0xffffffffu;
      }
      st.setreg(dst, static_cast<uint64_t>(mask));
      st.pc = next_pc;
      return 1;
    }

    if (op == "bar.sync" || op.rfind("bar.", 0) == 0) {
      // Block-wide barrier: the CTA scheduler keeps the thread parked until all
      // live threads of the block reach this point (see execute_kernel).
      st.pc = next_pc;
      return 2;
    }

    if (error) *error = "unsupported instruction '" + op + "' in " + fn.name;
    return 0;
}

}  // namespace

// ------------------------------------------------------------- entry point

bool execute_kernel(const PtxProgram& prog, int fnIndex,
                    const std::vector<uint8_t>& paramsBlob,
                    unsigned gx, unsigned gy, unsigned gz,
                    unsigned bx, unsigned by, unsigned bz,
                    std::string* error) {
  if (fnIndex < 0 || fnIndex >= static_cast<int>(prog.functions.size())) {
    if (error) *error = "kernel function index out of range";
    return false;
  }
  const PtxFunction& fn = prog.functions[fnIndex];

  // CTA-sized thread states for one block, stepped in lockstep (SIMT).
  const size_t threadsPerBlock = static_cast<size_t>(bz) * by * bx;

  for (uint64_t bz_i = 0; bz_i < gz; ++bz_i)
    for (uint64_t by_i = 0; by_i < gy; ++by_i)
      for (uint64_t bx_i = 0; bx_i < gx; ++bx_i) {
        // Per-block scratchpad for .shared variables (one per CTA).
        std::vector<uint8_t> shared(fn.sharedSize > 0 ? fn.sharedSize : 1, 0);
        const uintptr_t sharedBase = reinterpret_cast<uintptr_t>(shared.data());

        std::vector<ExecState> states(threadsPerBlock);
        std::vector<bool> alive(threadsPerBlock, true);
        std::vector<bool> blocked(threadsPerBlock, false);
        size_t n = 0;
        for (uint64_t tz = 0; tz < bz; ++tz)
          for (uint64_t ty = 0; ty < by; ++ty)
            for (uint64_t tx = 0; tx < bx; ++tx) {
              ExecState& st = states[n++];
              st.sharedBase = sharedBase;
              st.tid[0] = static_cast<uint32_t>(tx);
              st.tid[1] = static_cast<uint32_t>(ty);
              st.tid[2] = static_cast<uint32_t>(tz);
              st.ctaid[0] = static_cast<uint32_t>(bx_i);
              st.ctaid[1] = static_cast<uint32_t>(by_i);
              st.ctaid[2] = static_cast<uint32_t>(bz_i);
              st.ntid[0] = bx; st.ntid[1] = by; st.ntid[2] = bz;
              st.nctaid[0] = gx; st.nctaid[1] = gy; st.nctaid[2] = gz;
            }

        // Warp shuffles need cross-lane visibility: expose this block's states.
        g_block_states = &states;
        // CTA scheduler: all live threads step one instruction per round, so
        // shared-memory writes performed "before" a bar.sync are visible to the
        // threads that read "after" it (barrier release when every live thread
        // of the block has parked at one). This faithfully models SIMT memory
        // semantics; a step cap guards against runaway loops in the PTX.
        const uint64_t kMaxSteps = 100000000ull;
        uint64_t steps = 0;
        while (true) {
          bool anyAlive = false;
          for (size_t i = 0; i < threadsPerBlock; ++i) {
            if (!alive[i] || blocked[i]) continue;
            anyAlive = true;
            if (++steps > kMaxSteps) {
              if (error) *error = "emulation step limit exceeded (runaway loop?) in " + fn.name;
              g_block_states = nullptr;
              return false;
            }
            const int rc = run_thread_step(fn, paramsBlob, states[i], error);
            if (rc == 0) {
              g_block_states = nullptr;
              return false;
            }
            if (rc == 2) blocked[i] = true;
            else if (rc == 3) alive[i] = false;
          }
          if (!anyAlive) break;
          // All remaining live threads parked at a barrier -> release (bar.sync
          // met). Threads that already exited simply don't count.
          bool allBlocked = true;
          for (size_t i = 0; i < threadsPerBlock; ++i)
            if (alive[i] && !blocked[i]) { allBlocked = false; break; }
          if (allBlocked) {
            for (size_t i = 0; i < threadsPerBlock; ++i) blocked[i] = false;
            continue;
          }
        }
        g_block_states = nullptr;
      }
  return true;
}

}  // namespace rouge