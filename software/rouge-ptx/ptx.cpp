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
    if (!pred.empty() && pred[0] == '%') pred = pred.substr(1);
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

int align_up(int v, int a) { return (v + a - 1) / a * a; }

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
      inParams = (paren != std::string::npos);
      continue;
    }

    if (!cur) continue;  // pre-function directives: .version .target .address_size

    if (inParams) {
      if (s == ")") {
        inParams = false;
        continue;
      }
      if (s.rfind(".param", 0) == 0) {
        std::string ps = s;
        if (!ps.empty() && ps.back() == ',') ps.pop_back();
        std::stringstream pss(ps);
        std::string tok, type, name;
        pss >> tok;   // ".param"
        pss >> type;  // ".u64"
        pss >> name;
        const uint32_t w = param_width(type);
        if (w == 0) {
          if (error) *error = "unsupported param type '" + type + "' (line " + std::to_string(lineNo) + ")";
          return nullptr;
        }
        PtxParam p;
        p.name = name;
        p.width = static_cast<int>(w);
        p.index = static_cast<int>(cur->params.size());
        p.offset = cur->paramsBlobSize;
        cur->paramsBlobSize += static_cast<int>(w);
        cur->params.push_back(std::move(p));
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

// Resolve a memory operand: shared symbol first, otherwise a GPR holding an
// address (global, or shared address produced by cvta.to.shared).
uintptr_t mem_addr(const PtxFunction& fn, const ExecState& st,
                   const std::string& tok) {
  uintptr_t shared = 0;
  if (shared_symbol_addr(fn, st, tok, &shared)) return shared;
  return static_cast<uintptr_t>(st.reg(strip_brackets(tok)));
}

const PtxParam* find_param(const PtxFunction& fn, const std::string& name) {
  for (const auto& p : fn.params)
    if (p.name == name) return &p;
  return nullptr;
}

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

    if (op == "mul.wide.u32") {
      if (A.size() < 3) { if (error) *error = "bad mul.wide.u32 in " + fn.name; return 0; }
      const uint32_t a = static_cast<uint32_t>(operand_u64(st, A[1]));
      const uint32_t b = static_cast<uint32_t>(operand_u64(st, A[2]));
      st.setreg(A[0], static_cast<uint64_t>(a) * static_cast<uint64_t>(b));
      st.pc = next_pc;
      return 1;
    }

    if (op == "add.s64" || op == "add.u64" || op == "add.u32" || op == "add.s32") {
      if (A.size() < 3) { if (error) *error = "bad add in " + fn.name; return 0; }
      st.setreg(A[0], operand_u64(st, A[1]) + operand_u64(st, A[2]));
      st.pc = next_pc;
      return 1;
    }
    if (op == "sub.s64" || op == "sub.u64" || op == "sub.u32") {
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

    if (op.rfind("cvt.", 0) == 0) {
      // cvt.<dst>.<src> %d, %s  (integer conversions; a few float cases for realism)
      if (A.size() < 2) { if (error) *error = "bad cvt in " + fn.name; return 0; }
      std::string spec = op.substr(4);  // "<dst>.<src>"
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
      if (dst == "f32" && (src == "u32" || src == "s32" || src == "u16" || src == "s16")) {
        const int32_t sv = (src[0] == 'u') ? static_cast<int32_t>(raw32)
                                           : static_cast<int32_t>(raw);
        st.setreg(A[0], bit_cast_v<uint32_t>(static_cast<float>(sv)));
        st.pc = next_pc;
        return 1;
      }
      if (dst == "f32" && src == "u64") {
        st.setreg(A[0], bit_cast_v<uint32_t>(static_cast<float>(raw)));
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
      const uint32_t x = static_cast<uint32_t>(st.reg(A[1]));
      const uint32_t y = static_cast<uint32_t>(st.reg(A[2]));
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
      if (ty == "f32" || ty == "b32" || ty == "u32" || ty == "s32") {
        st.setreg(A[0], read_mem(reinterpret_cast<const uint8_t*>(addr), 4) & 0xffffffffu);
      } else if (ty == "u64" || ty == "b64" || ty == "s64") {
        st.setreg(A[0], read_mem(reinterpret_cast<const uint8_t*>(addr), 8));
      } else if (ty == "f16" || ty == "b16" || ty == "u16" || ty == "s16" ||
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
      if (ty == "f32") {
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
      // atom.add.{u32,s32,u64,s64,f32} %r, [addr], %val  -> old value
      // red.add.{...} [addr], %val                         -> no result
      // Scope/space qualifiers (".global", ".cta") are accepted and ignored:
      // memory is unified in both execution paths.
      if (op.find(".add.") == std::string::npos) {
        if (error) *error = "unsupported atomic '" + op + "' in " + fn.name;
        return 0;
      }
      const bool has_dst = op.rfind("atom.", 0) == 0;
      const size_t ai = has_dst ? 1 : 0;
      if (A.size() < ai + 2) { if (error) *error = "bad " + op + " in " + fn.name; return 0; }
      const std::string ty = op.substr(op.rfind('.') + 1);
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
              return false;
            }
            const int rc = run_thread_step(fn, paramsBlob, states[i], error);
            if (rc == 0) return false;
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
      }
  return true;
}

}  // namespace rouge