#include "ptx.h"

#include <atomic>
#include <cctype>
#include <cmath>
#include <limits>
#if defined(__linux__)
#include <sys/mman.h>
#endif
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

// PTX-идентификатор: буква, '_' или '$' первым символом, далее
// буквы/цифры/'_'/'$'. '$' — валидный первый символ: так nvcc 13
// (NVVM) пишет метки базовых блоков, например "$L__BB0_2:".
bool ident_start(char c) {
  return std::isalpha((unsigned char)c) || c == '_' || c == '$';
}

bool ident_cont(char c) {
  return std::isalnum((unsigned char)c) || c == '_' || c == '$';
}

// Поиск метки "IDENT:" в строке — в любом её месте. Возвращает индекс
// начала имени или npos; *colon — позиция ':', *label — имя метки.
// Двоеточие в PTX больше нигде не встречается (ни в опкодах, ни в
// типах, ни в операндах), поэтому первое вхождение однозначно метка.
size_t find_label(const std::string& s, size_t* colon, std::string* label) {
  for (size_t i = 0; i < s.size(); ++i) {
    if (!ident_start(s[i])) continue;
    size_t j = i + 1;
    while (j < s.size() && ident_cont(s[j])) ++j;
    if (j < s.size() && s[j] == ':') {
      *colon = j;
      *label = s.substr(i, j - i);
      return i;
    }
    i = j - 1;  // ++i в for перейдёт на j
  }
  return std::string::npos;
}

// Хвост строки после ':' метки: только пробелы и/или trailing-комментарий.
// ("//" уже срезан вызывающим кодом; блочный "/* ... */" — ещё нет.)
bool is_blank_or_comment(const std::string& s) {
  size_t i = 0;
  while (i < s.size() && std::isspace((unsigned char)s[i])) ++i;
  if (i == s.size()) return true;
  if (s.compare(i, 2, "/*") == 0) {
    const size_t end = s.find("*/", i + 2);
    if (end == std::string::npos) return false;
    i = end + 2;
    while (i < s.size() && std::isspace((unsigned char)s[i])) ++i;
    return i == s.size();
  }
  return false;
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
  std::string op;
  if (sp == std::string::npos) {
    op = rest;
  } else {
    op = rest.substr(0, sp);
  }
  // The .volatile qualifier only forbids caching/reordering across the
  // access; neither lowering path caches, so both are already effectively
  // volatile. Stripping it once in the parser keeps every consumer honest
  // without a qualifier check at each dispatch site.
  for (size_t v = op.find(".volatile"); v != std::string::npos;
       v = op.find(".volatile"))
    op.erase(v, 9);
  out->op = op;
  if (sp != std::string::npos) {
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
    // Trailing ',' (comma lists) and ';' (end of declaration, e.g. a lone
    // ".reg .b64 %SP;") are not part of the name.
    while (!t.empty() && (t.back() == ',' || t.back() == ';')) t.pop_back();
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
// A leading ".extern" is tolerated (nvcc puts dynamic .shared at module
// scope); an empty "[]" means dynamic shared: size fixed at launch, offset
// marks the dynamic region base after the static scratchpad.
void parse_shared_decl(const std::string& line, PtxFunction* fn) {
  std::string s = trim(line);
  const size_t semi = s.find(';');
  if (semi != std::string::npos) s = s.substr(0, semi);
  if (s.rfind(".extern", 0) == 0) s = trim(s.substr(7));
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
          if (rb == lb + 1) {
            v.dynamic = true;  // unsized "name[]": dynamic shared
            v.count = 0;
          } else {
            v.count = std::atoi(t.substr(lb + 1, rb - lb - 1).c_str());
          }
        }
      }
      if (v.name.empty()) continue;
      if (fn->sharedIndex.find(v.name) != fn->sharedIndex.end()) continue;
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

// Parse ".local .align 8 .b8 __local_depot0[120];": per-thread spill depot.
// Unlike .shared (one copy per block), each thread owns its copy; offset lays
// the variables out inside one per-thread depot of localSize bytes.
void parse_local_decl(const std::string& line, PtxFunction* fn) {
  std::string s = trim(line);
  const size_t semi = s.find(';');
  if (semi != std::string::npos) s = s.substr(0, semi);
  std::stringstream ss(s);
  std::string tok;
  int align = 0;
  int elemBytes = 1;
  ss >> tok;  // ".local"
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
      PtxFunction::PtxLocalVar v;
      const size_t lb = t.find('[');
      if (lb != std::string::npos) {
        const size_t rb = t.find(']', lb);
        if (rb != std::string::npos) {
          v.name = t.substr(0, lb);
          v.bytes = elemBytes * std::atoi(t.substr(lb + 1, rb - lb - 1).c_str());
        }
      } else {
        v.name = t;
        v.bytes = elemBytes;
      }
      if (v.name.empty()) continue;
      if (fn->localIndex.find(v.name) != fn->localIndex.end()) continue;
      v.align = align > 0 ? align : elemBytes;
      v.offset = align_up(fn->localSize, v.align);
      fn->localSize = v.offset + v.bytes;
      fn->localIndex[v.name] = static_cast<int>(fn->localVars.size());
      fn->localVars.push_back(std::move(v));
    }
  }
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

// ---- narrow float formats (e4m3 / e5m2 / e3m2 / e2m3 / e2m1) -----------------
// Bit layouts and special values are the OCP specs, not memory:
//   OFP8 (OCP 8-bit Floating Point Specification, Table 1-2):
//     e4m3: bias 7,  emax 8,  emin -6,  max 448 (0x7E), NaN only 0x7F/0xFF, no Inf;
//     e5m2: bias 15, emax 15, emin -14, max 57344 (0x7B), Inf 0x7C/0xFC.
//   OCP MX v1.0 (Table 7 element formats):
//     e3m2: bias 3, max 28, min normal 0.25; e2m3: bias 1, max 7.5, min normal 1;
//     e2m1: bias 1, max 6; none of the three has Inf or NaN.
//   PTX ISA 9.4, cvt section: conversions into these formats carry mandatory
//     .satfinite (clamp to sign-preserved MAX_NORM on overflow); NaN input
//     becomes NaN for e4m3/e5m2 destinations and positive MAX_NORM for
//     e3m2/e2m3/e2m1 destinations.
//
// Containers (PTX 5.2.3): e4m3/e5m2 live in a full .b8; e3m2/e2m3 in the lower
// 6 bits of a .b8 (upper 2 zero-padded); e2m1 in the central 4 bits [5:2] of a
// .b8 (2 zero bits on each side).

struct NarrowFpFormat {
  int expBits;     // E
  int mantBits;    // M
  int bias;        // exponent bias
  uint32_t maxBits;  // bit pattern of MAX_NORM (without sign), in the container
  uint32_t nanBits;  // bit pattern of canonical NaN (without sign), 0 = none
  int containerBits;  // 8 for all narrow formats here
  int fieldShift;     // bit offset of the field inside the container (e2m1: 2)
  int fieldWidth;     // width of the field (8, 6 or 4)
};

// Decode narrow bits (already extracted to the low fieldWidth bits) to float.
float narrow_to_f32(uint32_t bits, const NarrowFpFormat& F) {
  const uint32_t sign = (bits >> (F.fieldWidth - 1)) & 1u;
  const uint32_t exp = (bits >> F.mantBits) & ((1u << F.expBits) - 1u);
  const uint32_t man = bits & ((1u << F.mantBits) - 1u);
  const uint32_t fsign = sign << 31;
  const uint32_t maxExp = (1u << F.expBits) - 1u;
  uint32_t fbits;
  if (exp == 0) {
    if (man == 0) {
      fbits = fsign;  // +-0
    } else {
      // Subnormal: normalize, then rebias to binary32.
      uint32_t m = man;
      uint32_t shift = 0;
      while ((m & (1u << F.mantBits)) == 0) {
        m <<= 1;
        ++shift;
      }
      m &= (1u << F.mantBits) - 1u;
      const int fexp = 127 - F.bias + 1 - static_cast<int>(shift);
      const int fmanShift = 23 - F.mantBits;
      fbits = fsign | (static_cast<uint32_t>(fexp) << 23) |
              (m << fmanShift);
    }
  } else if (exp == maxExp) {
    // All-ones exponent. Whether this is Inf/NaN or the top normal binade
    // depends on the format: e4m3 reserves ONLY mantissa-all-ones as NaN
    // (exp==15, mant 0..6 are normals up to 448); e5m2 uses the whole binade
    // for Inf (mant 0) and NaN (mant 1..3); formats without NaN (e3m2/e2m3/
    // e2m1) are normal here by construction.
    const uint32_t allMant = (1u << F.mantBits) - 1u;
    const bool topNormal =
        ((F.maxBits >> F.mantBits) & maxExp) == maxExp;
    if (F.nanBits != 0 && (man == allMant || !topNormal)) {
      if (!topNormal && man == 0)
        fbits = fsign | 0x7f800000u;  // e5m2 Inf (the only Inf in this family)
      else
        fbits = fsign | 0x7fc00000u;  // NaN (payload not preserved: see encode)
    } else {
      fbits = fsign |
              (static_cast<uint32_t>(static_cast<int>(exp) - F.bias + 127) << 23) |
              (man << (23 - F.mantBits));
    }
  } else {
    fbits = fsign |
            (static_cast<uint32_t>(static_cast<int>(exp) - F.bias + 127) << 23) |
            (man << (23 - F.mantBits));
  }
  return bit_cast_v<float>(fbits);
}

// Encode float to narrow bits (low fieldWidth bits) with round-to-nearest-even
// and saturation to sign-preserved MAX_NORM on overflow (.satfinite).
// nanToMax: true for e3m2/e2m3/e2m1 (NaN -> positive MAX_NORM per PTX cvt),
// false for e4m3/e5m2 (NaN -> sign-preserved NaN).
uint32_t f32_to_narrow(float f, const NarrowFpFormat& F, bool roundToZero,
                       bool nanToMax) {
  const uint32_t x = bit_cast_v<uint32_t>(f);
  const uint32_t sign = (x >> 31) & 1u;
  const uint32_t e = (x >> 23) & 0xffu;
  const uint32_t man = x & 0x7fffffu;
  const uint32_t maxExp = (1u << F.expBits) - 1u;
  // NaN input.
  if (e == 0xffu && man != 0) {
    if (nanToMax) return F.maxBits;  // positive MAX_NORM (PTX cvt rule)
    return (sign << (F.fieldWidth - 1)) | F.nanBits;
  }
  // Inf input saturates (satfinite is mandatory on these conversions).
  if (e == 0xffu) return (sign << (F.fieldWidth - 1)) | F.maxBits;
  // Signed zero is exactly representable: preserve the sign (the range logic
  // below would drop it to +0).
  if (e == 0 && man == 0) return sign << (F.fieldWidth - 1);
  const int unbiased = static_cast<int>(e) - 127;
  // Target biased exponent for a normal result.
  const int tExp = unbiased + F.bias;
  const int M = F.mantBits;
  // Overflow: only tExp PAST the top binade saturates up front. tExp inside
  // the top binade rounds normally below: e4m3's exp==15 holds normals up to
  // 0x7E (only mantissa-all-ones is NaN), so pre-saturating it would corrupt
  // exactly-representable values like 256.0.
  if (tExp > static_cast<int>(maxExp))
    return (sign << (F.fieldWidth - 1)) | F.maxBits;
  if (tExp <= 0) {
    // Subnormal or underflow. Full significand with implicit 1, shifted.
    if (tExp < -M) return 0;  // underflows past min-subnormal/2 (sign dropped: +0)
    // Hmm: tExp in [1-M, 0]: value = 2^(1-bias) * (mantissa >> shift).
    const uint32_t full = (1u << 23) | man;  // 24-bit significand
    const int shift = 1 - tExp + (23 - M);
    uint32_t val = full >> shift;
    const uint32_t rem = full & ((shift >= 32) ? 0xffffffffu : ((1u << shift) - 1u));
    const uint32_t half = 1u << (shift - 1);
    if (!roundToZero && (rem > half || (rem == half && (val & 1u)))) ++val;
    // Rounding a subnormal up can carry into the normal range; the result
    // still fits (exp field 1 at most here... cap defensively at maxBits).
    const uint32_t cap = (1u << (F.expBits + M)) - 1u;
    if (val > cap) val = cap;
    if (F.nanBits != 0 && (val >> M) == maxExp) {
      // Carried into the all-ones exponent: e5m2 subnormal->tiny normal is
      // fine (mant 0 => Inf? no: subnormal rounding up gives exp=1,mant=0,
      // the smallest normal). Only e4m3's exp==maxExp is dangerous, and a
      // subnormal can never round that far. Unreachable; kept for symmetry.
      return (sign << (F.fieldWidth - 1)) | F.maxBits;
    }
    return (sign << (F.fieldWidth - 1)) | (val & cap);
  }
  // Normal range: keep top M bits, round the rest.
  const int dropBits = 23 - M;
  uint32_t val = (static_cast<uint32_t>(tExp) << M) | (man >> dropBits);
  const uint32_t rem = man & ((1u << dropBits) - 1u);
  const uint32_t half = 1u << (dropBits - 1);
  if (!roundToZero && (rem > half || (rem == half && (val & 1u)))) ++val;
  // Rounding may carry into (or inside) the all-ones exponent. Whether that
  // saturates depends on the format, mirroring the decode rule above:
  // e4m3 saturates only on the NaN pattern (exp==15, mantissa all ones) —
  // this is exactly the OFP8 "448..464 rounds to 448" rule, generalized;
  // e5m2 saturates on anything in exp==31 (Inf or NaN out); no-NaN formats
  // saturate only past the top binade.
  {
    const uint32_t gotExp = val >> M;
    const uint32_t allMant = (1u << M) - 1u;
    const bool topNormal =
        ((F.maxBits >> F.mantBits) & maxExp) == maxExp;
    if (gotExp > maxExp) return (sign << (F.fieldWidth - 1)) | F.maxBits;
    if (gotExp == maxExp && F.nanBits != 0) {
      if (!topNormal) return (sign << (F.fieldWidth - 1)) | F.maxBits;
      if ((val & allMant) == allMant)
        return (sign << (F.fieldWidth - 1)) | F.maxBits;
    }
    return (sign << (F.fieldWidth - 1)) |
           (val & ((1u << (F.expBits + M)) - 1u));
  }
}

// Place field bits into / extract from the .b8 container.
uint32_t narrow_pack(uint32_t field, const NarrowFpFormat& F) {
  const uint32_t mask = (F.fieldWidth == 32) ? 0xffffffffu : ((1u << F.fieldWidth) - 1u);
  return (field & mask) << F.fieldShift;
}
uint32_t narrow_unpack(uint32_t container, const NarrowFpFormat& F) {
  const uint32_t mask = (F.fieldWidth == 32) ? 0xffffffffu : ((1u << F.fieldWidth) - 1u);
  return (container >> F.fieldShift) & mask;
}

// maxBits — битовый образ MAX_NORM БЕЗ знака (в поле формата): знаковый бит
// обязан быть нулём, иначе насыщение при положительном переполнении
// затирает знак результата и даёт -MAX вместо +MAX (поймано пробой
// satfinite для e3m2/e2m3/e2m1: 1000.0 -> 0x3F = -28 вместо 0x1F = +28).
const NarrowFpFormat kE4M3 = {4, 3, 7, 0x7Eu, 0x7Fu, 8, 0, 8};
const NarrowFpFormat kE5M2 = {5, 2, 15, 0x7Bu, 0x7Fu, 8, 0, 8};
const NarrowFpFormat kE3M2 = {3, 2, 3, 0x1Fu, 0x00u, 8, 0, 6};
const NarrowFpFormat kE2M3 = {2, 3, 1, 0x1Fu, 0x00u, 8, 0, 6};
const NarrowFpFormat kE2M1 = {2, 1, 1, 0x07u, 0x00u, 8, 2, 4};

}  // namespace

// ---------------------------------------------------------------- parser

std::unique_ptr<PtxProgram> parse_ptx(const std::string& text, std::string* error) {
  auto prog = std::make_unique<PtxProgram>();
  std::stringstream ss(text);
  std::string line;
  PtxFunction* cur = nullptr;
  bool inParams = false;
  int lineNo = 0;
  // Module-scope ".extern .shared" lines (dynamic shared, nvcc puts them
  // before the entries): stashed here, attached to every function below.
  std::vector<std::string> externShared;
  // Module-scope static .shared/.local (nvcc 13 emits kernel __shared__ arrays
  // outside .entry); attached to every function after the body loop.
  std::vector<std::string> moduleShared;
  std::vector<std::string> moduleLocal;

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
      // ".target sm_75" records the compute capability the module was built
      // for (needed for cudaFuncAttributes' ptxVersion/binaryVersion).
      if (s.rfind(".target", 0) == 0) {
        const size_t sm = s.find("sm_");
        if (sm != std::string::npos)
          prog->targetSm = std::atoi(s.c_str() + sm + 3);
      }
      if (s.rfind(".global", 0) == 0 || s.rfind(".const", 0) == 0)
        parse_global_decl(s, prog.get());
      // ".extern .shared ..." (dynamic shared) applies to every kernel in
      // the module; stashed and attached to each function after the loop.
      if (s.rfind(".extern", 0) == 0 && s.find(".shared") != std::string::npos)
        externShared.push_back(s);
      // nvcc 13 declares kernel-scope static __shared__ arrays at MODULE
      // scope, outside any .entry ("_ZZ...E2As[1024]" for matrixMul). Such a
      // declaration still belongs to every kernel of the module, so stash the
      // line and hand it to each function once the body is closed. Dropping
      // it silently left the symbol unresolved: addresses resolved to zero,
      // every store hit one cell and loops such as matrixMul's tile fill
      // never terminated.
      if (s.rfind(".shared", 0) == 0) moduleShared.push_back(s);
      if (s.rfind(".local", 0) == 0) moduleLocal.push_back(s);
      // ".extern .global/.const ..." (nvcc emits these for unused C++ markers):
      // record like a normal global so a use resolves to a real external
      // symbol (link/JIT fails loudly if nobody defines it) instead of
      // silently reading address zero.
      if (s.rfind(".extern", 0) == 0 &&
          (s.find(".global") != std::string::npos ||
           s.find(".const") != std::string::npos) &&
          s.find(".shared") == std::string::npos)
        parse_global_decl(s.substr(7), prog.get());
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
    if (s.rfind(".local", 0) == 0) {
      parse_local_decl(s, cur);
      continue;
    }
    if (!s.empty() && s[0] == '.') continue;  // .align/... directives

    // Метка: "NAME:" — в любом месте строки, с необязательным
    // trailing-комментарием; текст до метки (если есть) —
    // инструкция, после двоеточия — пусто или комментарий.
    // nvcc 13 пишет метки вида "$L__BB0_2:" ('$' — валидный
    // символ идентификатора, см. find_label).
    size_t colon = std::string::npos;
    std::string label;
    const size_t lstart = find_label(s, &colon, &label);
    if (lstart != std::string::npos &&
        is_blank_or_comment(s.substr(colon + 1))) {
      const std::string head = trim(s.substr(0, lstart));
      if (!head.empty()) {
        // Инструкция и метка в одной строке: инструкция выдаётся
        // первой, чтобы метка указывала на следующую за ней.
        PtxInstruction instr;
        if (!parse_instruction(head, &instr)) {
          if (error) *error = "cannot parse instruction (line " + std::to_string(lineNo) + "): " + head;
          return nullptr;
        }
        cur->body.push_back(std::move(instr));
      }
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

  // Attach stashed module-scope ".extern .shared" decls to every function:
  // dynamic region base lands after each kernel's static scratchpad.
  for (const std::string& decl : externShared)
    for (auto& fn : prog->functions) parse_shared_decl(decl, &fn);
  // Module-scope static .shared/.local from nvcc 13 (see the comment at the
  // collection site): these belong to every kernel in the module, and their
  // laid-out offsets must be assigned AFTER the function's own declarations
  // so the per-kernel scratchpad layout stays contiguous.
  for (const std::string& decl : moduleShared)
    for (auto& fn : prog->functions) parse_shared_decl(decl, &fn);
  for (const std::string& decl : moduleLocal)
    for (auto& fn : prog->functions) parse_local_decl(decl, &fn);

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
  std::vector<uint8_t> local;  // per-thread .local depot (spill slots)
  uintptr_t localBase = 0;     // host address of this thread's depot

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

// Hardware cycle/id registers (no host equivalent; notably %clock64: CPU
// wall time is not GPU cycles, so emulation would silently lie about units).
bool is_hw_sreg(const std::string& n) {
  return n == "clock" || n == "clock64" || n == "warpid" || n == "nwarpid" ||
         n == "smid" || n == "nsmid" || n == "gridid" || n == "laneid";
}

uint64_t read_mem(const uint8_t* p, int width) {
  uint64_t v = 0;
  std::memcpy(&v, p, static_cast<size_t>(width));
  return v;
}

// Parse a PTX float literal into its IEEE bit pattern. PTX writes floats as
// "0f3F800000" (raw hex of the 32-bit pattern) or "0d3FF0000000000000"
// (f64), and sometimes as a decimal value ("1f3f800000" is never seen; plain
// decimal floats do not appear in nvcc output but "1.0" does appear in some
// hand-written PTX). Returns false when tok is not a float literal.
bool parse_float_literal(const std::string& tok, uint64_t* bits, int* width) {
  if (tok.size() < 3) return false;
  const char kind = tok[1];
  if (kind != 'f' && kind != 'F' && kind != 'd' && kind != 'D') return false;
  const std::string digits = tok.substr(2);
  for (char c : digits)
    if (!std::isxdigit(static_cast<unsigned char>(c))) return false;
  if (digits.empty()) return false;
  if (kind == 'f' || kind == 'F') {
    *bits = std::strtoull(digits.c_str(), nullptr, 16);
    *width = 32;
    return true;
  }
  *bits = std::strtoull(digits.c_str(), nullptr, 16);
  *width = 64;
  return true;
}

// Read an operand that may be a special register, an immediate literal, or a GPR.
uint64_t operand_u64(const ExecState& st, const std::string& tok) {
  uint64_t v = 0;
  if (st.spec_reg(tok, &v)) return v;
  if (!tok.empty() && (std::isdigit(static_cast<unsigned char>(tok[0])) || tok[0] == '-')) {
    // "0f3F800000" is a float literal, not the integer 0: strtoull would
    // stop at the 'f' and silently yield 0 (vendor PTX is full of these).
    uint64_t fbits = 0;
    int fw = 0;
    if (parse_float_literal(tok, &fbits, &fw) && fw == 32) return fbits;
    return static_cast<uint64_t>(std::strtoull(tok.c_str(), nullptr, 10));
  }
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

// If the operand names a .local variable (bare or with a +/-Const byte
// suffix), return its host depot address. Each thread owns its depot copy.
bool local_symbol_addr(const PtxFunction& fn, const ExecState& st,
                       const std::string& tok, uintptr_t* out) {
  std::string inner = strip_brackets(tok);
  int64_t constOff = 0;
  std::string base;
  int64_t off = 0;
  if (split_reg_offset(inner, &base, &off)) {
    inner = base;
    constOff = off;
  }
  const auto it = fn.localIndex.find(inner);
  if (it == fn.localIndex.end()) return false;
  if (constOff < 0) return false;
  *out = static_cast<uintptr_t>(st.localBase) +
         static_cast<uintptr_t>(fn.localVars[it->second].offset) +
         static_cast<uintptr_t>(constOff);
  return true;
}

// Module-global backing storage for the current launch (name -> host
// address), installed by execute_kernel from its globalBases argument.
// Same single-threaded pattern as g_block_states below: the emulator
// steps one CTA at a time, so a static is safe here.
static const std::unordered_map<std::string, uint64_t>* g_global_bases = nullptr;

// A module-scope .global/.const symbol ("sym", "[sym]") resolves to its
// backing-store address; other operands fall through (returns false).
bool module_global_addr(const std::string& tok, uint64_t* out) {
  if (g_global_bases == nullptr || g_global_bases->empty()) return false;
  std::string name = strip_brackets(tok);
  // A symbol with an offset ("[sym+8]") is not a bare symbol.
  if (name.find('+') != std::string::npos || name.find('-') != std::string::npos)
    return false;
  const auto it = g_global_bases->find(name);
  if (it == g_global_bases->end()) return false;
  *out = it->second;
  return true;
}

// Resolve a memory operand: shared symbol first, then a shared symbol with a
// constant offset, then a GPR holding an address (global, or a shared address
// produced by cvta.to.shared), then a GPR plus a constant offset.
uintptr_t mem_addr(const PtxFunction& fn, const ExecState& st,
                   const std::string& tok) {
  uintptr_t shared = 0;
  if (shared_symbol_addr(fn, st, tok, &shared)) return shared;
  uintptr_t local = 0;
  if (local_symbol_addr(fn, st, tok, &local)) return local;
  // Direct "[sym]" / "[sym+off]" over a module-scope .global/.const.
  // (The common nvcc pattern goes through cvta/mov into a register
  // first, which is covered above; this is the direct form.)
  uint64_t gbase = 0;
  if (module_global_addr(tok, &gbase)) return static_cast<uintptr_t>(gbase);
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
      // Host virtual addresses in emulation -> identity, except shared/local
      // symbols which resolve to the block scratchpad / thread depot address.
      if (A.size() >= 2) {
        uintptr_t shared = 0;
        if (shared_symbol_addr(fn, st, A[1], &shared))
          st.setreg(A[0], static_cast<uint64_t>(shared));
        else {
          uintptr_t local = 0;
          if (local_symbol_addr(fn, st, A[1], &local))
            st.setreg(A[0], static_cast<uint64_t>(local));
          else {
            // Module-scope .global/.const: backing store allocated by the
            // API layer (cuModuleGetGlobal); without it the old zero stands.
            uint64_t gbase = 0;
            if (module_global_addr(A[1], &gbase))
              st.setreg(A[0], gbase);
            else {
              uint64_t v = 0;
              if (!st.spec_reg(A[1], &v)) v = st.reg(A[1]);
              st.setreg(A[0], v);
            }
          }
        }
      }
      st.pc = next_pc;
      return 1;
    }

    if (op.rfind("mov.", 0) == 0) {
      if (A.size() < 2) { if (error) *error = "bad mov in " + fn.name; return 0; }
      // Hardware cycle/id registers have no host equivalent: refuse loudly
      // instead of reading an uninitialised zero (clock.ptx caught this).
      if (is_hw_sreg(A[1])) {
        if (error)
          *error = "unsupported mov of %" + A[1] + " in the interpreter in " +
                   fn.name + " (no host equivalent of this hardware register)";
        return 0;
      }
      // "mov.b64 %SPL, __local_depot0": materialise the thread depot address.
      // A module-scope .global/.const source resolves to its backing store
      // (same rule as cvta above); anything else is a register/immediate.
      //
      // "mov.u32 %r11, smem_symbol" (nvcc 13 demoted-shared form) also
      // materialises the scratchpad ADDRESS into a 32-bit register: the
      // symbol must win over the register lookup, otherwise the register
      // silently reads 0 and every thread of the block stores to address
      // zero (reduction's tree loop then never converges).
      uintptr_t shared = 0;
      uintptr_t local = 0;
      uint64_t gbase = 0;
      const auto dst_is32 = op == "mov.u32" || op == "mov.s32" || op == "mov.b32";
      // A 32-bit destination keeps the low 32 bits of the address, exactly as
      // the hardware and the AOT path do (fit_reg / width_mask). This is only
      // sound because the interpreter's scratchpad lives in low memory
      // (see scratchpad_base below), so base + in-bounds offset never crosses
      // 4GB — the same invariant the AOT runtime establishes with MAP_32BIT.
      const auto put_addr = [&](uint64_t a) {
        st.setreg(A[0], dst_is32 ? static_cast<uint64_t>(static_cast<uint32_t>(a)) : a);
      };
      if (shared_symbol_addr(fn, st, A[1], &shared))
        put_addr(static_cast<uint64_t>(shared));
      else if (local_symbol_addr(fn, st, A[1], &local))
        put_addr(static_cast<uint64_t>(local));
      else if (module_global_addr(A[1], &gbase))
        put_addr(gbase);
      else if (dst_is32)
        st.setreg(A[0], static_cast<uint64_t>(static_cast<uint32_t>(operand_u64(st, A[1]))));
      else
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
      const uint64_t r = operand_u64(st, A[1]) + operand_u64(st, A[2]);
      // A 32-bit form wraps at 32 bits (width_mask in the AOT path does the
      // same); leaving the high bits would let a sign/zero-extended 32-bit
      // value leak into an address computation.
      st.setreg(A[0], op.back() == '2' ? width_mask(r, 32) : r);
      st.pc = next_pc;
      return 1;
    }
    if (op == "sub.s64" || op == "sub.u64" || op == "sub.u32" || op == "sub.s32") {
      if (A.size() < 3) { if (error) *error = "bad sub in " + fn.name; return 0; }
      const uint64_t r = operand_u64(st, A[1]) - operand_u64(st, A[2]);
      st.setreg(A[0], op.back() == '2' ? width_mask(r, 32) : r);
      st.pc = next_pc;
      return 1;
    }
    if (op == "add.f32" || op == "sub.f32" || op == "mul.f32" || op == "div.f32" ||
        op == "fma.rn.f32") {
      // fma.rn.f32 — fused multiply-add с одним округлением: nvcc 13
      // эмитит его в обычных циклах накопления (mmul поймал отсутствие).
      // std::fma даёт то же одно округление, что железный FMA и llvm.fma.
      const size_t need = op[0] == 'f' ? 4 : 3;
      if (A.size() < need) { if (error) *error = "bad fp op in " + fn.name; return 0; }
      const float x = bit_cast_v<float>(static_cast<uint32_t>(operand_u64(st, A[1])));
      const float y = bit_cast_v<float>(static_cast<uint32_t>(operand_u64(st, A[2])));
      float r = 0.0f;
      if (op == "add.f32") r = x + y;
      else if (op == "sub.f32") r = x - y;
      else if (op == "mul.f32") r = x * y;
      else if (op == "fma.rn.f32")
        r = std::fma(x, y,
                     bit_cast_v<float>(static_cast<uint32_t>(operand_u64(st, A[3]))));
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

    // ---- dp4a / dp2a: dot product of byte pairs ---------------------------
    // nvcc emits dp4a for int8 dot products (the whole point of int8 GEMM),
    // so real PTX needs it. Semantics (PTX ISA 9.7.1.24/9.7.1.25):
    //   dp4a.atype.btype d, a, b, c:
    //     Va = the four bytes of a (sign/zero extended per .atype),
    //     Vb = the four bytes of b (per .btype),
    //     d = c + sum_i Va[i] * Vb[i].
    //   dp2a.mode.atype.btype d, a, b, c:
    //     Va = the two 16-bit halves of a (per .atype),
    //     Vb = the four bytes of b (per .btype),
    //     b_select = (mode == .lo) ? 0 : 2,
    //     d = c + Va[0]*Vb[b_select] + Va[1]*Vb[b_select+1].
    if (op.rfind("dp4a.", 0) == 0 || op.rfind("dp2a.", 0) == 0) {
      if (A.size() < 4) { if (error) *error = "bad " + op + " in " + fn.name; return 0; }
      const uint64_t a = operand_u64(st, A[1]);
      const uint64_t b = operand_u64(st, A[2]);
      const uint64_t c = operand_u64(st, A[3]);
      const bool wide = op.rfind("dp4a.", 0) == 0;
      // .mode applies to dp2a only: .lo uses the low byte pair of b.
      const bool use_low = wide || (op.find(".lo.") != std::string::npos);
      // .atype applies to a, .btype to b. Token layout:
      //   dp4a.atype.btype  |  dp2a.mode.atype.btype
      std::vector<std::string> tok;
      {
        std::string cur;
        for (char ch : op) {
          if (ch == '.') { tok.push_back(cur); cur.clear(); }
          else cur.push_back(ch);
        }
        tok.push_back(cur);
      }
      const size_t at_idx = wide ? 1u : 2u;   // индекс .atype в tok
      const std::string atype = tok.size() > at_idx ? tok[at_idx] : std::string("u32");
      const std::string btype = tok.size() > at_idx + 1u ? tok[at_idx + 1u] : std::string("u32");
      const bool a_signed = atype == "s32";
      const bool b_signed = btype == "s32";
      int64_t sum = 0;
      if (wide) {
        for (int i = 0; i < 4; ++i) {
          const uint64_t am = (a >> (8 * i)) & 0xffull;
          const uint64_t bm = (b >> (8 * i)) & 0xffull;
          const int64_t av = a_signed ? (int64_t)(int8_t)am : (int64_t)am;
          const int64_t bv = b_signed ? (int64_t)(int8_t)bm : (int64_t)bm;
          sum += av * bv;
        }
      } else {
        for (int i = 0; i < 2; ++i) {
          const uint64_t am = (a >> (16 * i)) & 0xffffull;
          const int64_t av = a_signed ? (int64_t)(int16_t)am : (int64_t)am;
          const int bi = (use_low ? 0 : 2) + i;
          const uint64_t bm = (b >> (8 * bi)) & 0xffull;
          const int64_t bv = b_signed ? (int64_t)(int8_t)bm : (int64_t)bm;
          sum += av * bv;
        }
      }
      const uint64_t r = static_cast<uint64_t>(c) + static_cast<uint64_t>(sum);
      st.setreg(A[0], width_mask(r, 32));
      st.pc = next_pc;
      return 1;
    }

    // ---- prmt.b32: byte permute -------------------------------------------
    // nvcc uses prmt to pack bytes (four byte loads into one 32-bit store is
    // the typical case), so real PTX needs it — otherwise even a trivial
    // int8 kernel cannot be translated.
    // Semantics (PTX ISA 9.7.10.7): the eight source bytes are the 64-bit
    // value {b, a} counted from the low end, i.e. index 0..3 are a's bytes
    // and 4..7 are b's. Each 4-bit field of c selects the source byte for
    // d.b0..d.b3 (field 0 -> d.b0); the field's msb asks for the byte's SIGN
    // to be replicated over the whole 8-bit target position instead of the
    // literal byte.
    if (op.rfind("prmt.", 0) == 0) {
      if (A.size() < 4) { if (error) *error = "bad " + op + " in " + fn.name; return 0; }
      const uint32_t a = static_cast<uint32_t>(operand_u64(st, A[1]));
      const uint32_t b = static_cast<uint32_t>(operand_u64(st, A[2]));
      const uint32_t c = static_cast<uint32_t>(operand_u64(st, A[3]));
      // The mode field (c bits 19:16) must be zero; the other modes have
      // different semantics and are refused loudly rather than guessed.
      if (((c >> 16) & 0xfu) != 0u) {
        if (error) *error = "unsupported prmt mode in " + fn.name;
        return 0;
      }
      const uint64_t src = (uint64_t)a | ((uint64_t)b << 32);
      uint32_t d = 0;
      for (int i = 0; i < 4; ++i) {
        const unsigned field = (c >> (4 * i)) & 0xfu;
        const unsigned sel = field & 0x7u;
        const uint32_t byte = (uint32_t)((src >> (sel * 8)) & 0xffu);
        // msb of the field: replicate the sign of the selected byte.
        const uint32_t val = (field & 0x8u) ? ((byte & 0x80u) ? 0xffu : 0x00u) : byte;
        d |= val << (8 * i);
      }
      st.setreg(A[0], width_mask(d, 32));
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

    // f16x2/bf16x2 fallback for non-memory ops: payload as i32. Matched on the
    // full ".f16x2"/".bf16x2" type names, never on a bare "x2": the latter also
    // occurs in matrix-instruction multiplicities (ldmatrix.x2, stmatrix.x2),
    // which this fallback used to swallow silently — every .x2 matrix move
    // executed as a no-op bit-copy with zeroed destinations (caught by the
    // mma_pipe kernel, the first .x2 user; .x1/.x4 never contain "x2").
    if ((op.find(".f16x2") != std::string::npos ||
         op.find(".bf16x2") != std::string::npos) &&
        op.rfind("ld.", 0) != 0 && op.rfind("st.", 0) != 0) {
      // f16x2/bf16x2 fallback for non-memory ops: payload as i32.
      // Narrow-float cvt forms (e4m3x2/e5m2x2/...) are NOT bit-copies: they
      // fall through to the cvt block below, which implements RNE+satfinite.
      const bool narrowCvt =
          op.find(".e4m3x2") != std::string::npos ||
          op.find(".e5m2x2") != std::string::npos ||
          op.find(".e3m2x2") != std::string::npos ||
          op.find(".e2m3x2") != std::string::npos ||
          op.find(".e2m1x2") != std::string::npos;
      if (!narrowCvt && A.size() >= 2) {
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

      // Saturating and explicitly-rounded conversions (cvt.sat.*, cvt.rz/ru/rp/
      // rm/rzi.*) are deliberately not emulated: they need PTX's saturation
      // and rounding rules, so fail loudly instead of silently handing back
      // the source bits. ".rn" stays legal for the f16/bf16 narrowing handled
      // below, for int->float (round-to-nearest-even IS the default there),
      // and for the packed narrow-float destinations (e4m3x2/e5m2x2/...),
      // whose RNE+satfinite semantics are implemented in the packed block.
      // ".rz" is the truncating counterpart there. ".satfinite" is mandatory
      // on those packed forms and means clamp-to-MAX_NORM (never Inf/NaN
      // out). ".relu" clamps negatives to +0. Refused: .rs (needs an rbits
      // operand with Figure-42 layout), .scaled (needs a scale factor),
      // .pzo/.ftz, and the x4 packed forms.
      const bool narrowDst =
          op.find(".e4m3x2") != std::string::npos ||
          op.find(".e5m2x2") != std::string::npos ||
          op.find(".e3m2x2") != std::string::npos ||
          op.find(".e2m3x2") != std::string::npos ||
          op.find(".e2m1x2") != std::string::npos;
      {
        std::stringstream mods(full);
        std::string mt;
        while (std::getline(mods, mt, '.')) {
          const bool is_round = mt == "rn" || mt == "rz" || mt == "rm" ||
                                mt == "rp" || mt == "ru" || mt == "rzi";
          if (mt != "sat" && mt != "satfinite" && mt != "relu" && !is_round) {
            // Unknown tokens here are either type names (e4m3x2, f32 — fine,
            // skip) or modifiers we do NOT implement: those must refuse
            // loudly, never be silently ignored (a skipped .ftz would
            // corrupt every subnormal; a skipped .rs would change rounding).
            const bool refused = mt == "rs" || mt == "rna" || mt == "ftz" ||
                                 mt == "pzo" || mt.find("scaled") != std::string::npos ||
                                 mt.find(':') != std::string::npos;
            if (!refused) continue;
          } else {
            if (mt == "relu") continue;  // always just a clamp, handled below
            if (mt == "satfinite" && narrowDst) continue;
            if ((mt == "rn" || mt == "rz") && narrowDst) continue;
            if (mt == "rn" &&
                (op.find(".f16") != std::string::npos ||
                 op.find(".bf16") != std::string::npos ||
                 // int -> float, any width: RNE is the only defined behaviour
                 op.find(".f32") != std::string::npos ||
                 op.find(".f64") != std::string::npos))
              continue;
          }
          if (error)
            *error = "unsupported rounding/saturating conversion '" + op +
                     "' in " + fn.name;
          return 0;
        }
      }

      std::string spec = full;
      // Drop ALL leading rounding/saturation modifiers so
      // "rn.satfinite.e4m3x2.f32" parses as dst="e4m3x2", src="f32" (each
      // modifier was validated above).
      for (;;) {
        const size_t firstDot = spec.find('.');
        const std::string head =
            firstDot == std::string::npos ? spec : spec.substr(0, firstDot);
        if (head == "rn" || head == "rz" || head == "rm" || head == "rp" ||
            head == "ru" || head == "rzi" || head == "sat" || head == "satfinite" ||
            head == "relu" || head == "ftz" || head == "pzo")
          spec = firstDot == std::string::npos ? std::string() : spec.substr(firstDot + 1);
        else
          break;
      }
      const size_t dot = spec.find('.');
      const std::string dst = dot == std::string::npos ? spec : spec.substr(0, dot);
      const std::string src = dot == std::string::npos ? "" : spec.substr(dot + 1);
      const uint64_t raw = operand_u64(st, A[1]);
      const uint32_t raw32 = static_cast<uint32_t>(raw);

      // Packed narrow-float conversions (PTX 9.4 cvt section; the forms nvcc
      // 13 actually emits, verified on cuda_fp8.h output):
      //   cvt.{rn,rz}.satfinite.e4m3x2.f32  d, a, b   (d.b16: a->upper, b->lower)
      //   cvt.rn.f16x2.e4m3x2               d, a      (d=2xf16: upper->upper...)
      // and the e5m2/e3m2/e2m3/e2m1 x bf16x2 analogues. .relu clamps negatives
      // to +0 (NaN -> canonical NaN for e4m3/e5m2, positive MAX_NORM where the
      // format has no NaN). x4/.rs/.scaled forms refuse loudly above.
      const NarrowFpFormat* narrowFmt = nullptr;
      bool narrowNanToMax = false;
      if (dst == "e4m3x2") { narrowFmt = &kE4M3; narrowNanToMax = false; }
      else if (dst == "e5m2x2") { narrowFmt = &kE5M2; narrowNanToMax = false; }
      else if (dst == "e3m2x2") { narrowFmt = &kE3M2; narrowNanToMax = true; }
      else if (dst == "e2m3x2") { narrowFmt = &kE2M3; narrowNanToMax = true; }
      else if (dst == "e2m1x2") { narrowFmt = &kE2M1; narrowNanToMax = true; }
      if (narrowFmt != nullptr && src == "f32") {
        if (A.size() < 3) { if (error) *error = "bad " + op + " in " + fn.name; return 0; }
        const bool rz = op.find(".rz.") != std::string::npos;
        const bool relu = op.find(".relu") != std::string::npos;
        const auto convOne = [&](uint32_t bits) -> uint32_t {
          float v = bit_cast_v<float>(bits);
          if (relu) {
            if (std::isnan(v))
              return narrowNanToMax ? narrow_pack(narrowFmt->maxBits, *narrowFmt)
                                    : narrow_pack(narrowFmt->nanBits, *narrowFmt);
            if (!(v >= 0.0f)) return 0;  // negative (incl. -0) -> +0
            v = v;  // keep value, fall through to normal encode
          }
          return narrow_pack(f32_to_narrow(v, *narrowFmt, rz, narrowNanToMax),
                             *narrowFmt);
        };
        const uint32_t hi =
            convOne(static_cast<uint32_t>(operand_u64(st, A[1])));
        const uint32_t lo =
            convOne(static_cast<uint32_t>(operand_u64(st, A[2])));
        st.setreg(A[0], (hi << 8) | lo);
        st.pc = next_pc;
        return 1;
      }
      const NarrowFpFormat* narrowSrc = nullptr;
      if (src == "e4m3x2") narrowSrc = &kE4M3;
      else if (src == "e5m2x2") narrowSrc = &kE5M2;
      else if (src == "e3m2x2") narrowSrc = &kE3M2;
      else if (src == "e2m3x2") narrowSrc = &kE2M3;
      else if (src == "e2m1x2") narrowSrc = &kE2M1;
      if (narrowSrc != nullptr && (dst == "f16x2" || dst == "bf16x2")) {
        const bool toBf16 = (dst == "bf16x2");
        const bool relu = op.find(".relu") != std::string::npos;
        const auto convOne = [&](uint32_t packed) -> uint32_t {
          float v = narrow_to_f32(narrow_unpack(packed, *narrowSrc), *narrowSrc);
          if (relu) {
            if (std::isnan(v)) v = std::numeric_limits<float>::quiet_NaN();
            else if (!(v >= 0.0f)) v = 0.0f;
          }
          return toBf16 ? f32_to_bf16_rne(v) : f32_to_f16_rne(v);
        };
        const uint32_t hi = convOne((raw32 >> 8) & 0xffu);
        const uint32_t lo = convOne(raw32 & 0xffu);
        st.setreg(A[0], (hi << 16) | lo);
        st.pc = next_pc;
        return 1;
      }

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
      const uint64_t xu = operand_u64(st, A[1]);
      const uint64_t yu = operand_u64(st, A[2]);
      const size_t d1 = op.find('.');
      std::string cond = (d1 == std::string::npos) ? "ge" : op.substr(d1 + 1);
      const size_t d2 = cond.find('.');
      const std::string ty = (d2 == std::string::npos) ? std::string() : cond.substr(d2 + 1);
      if (d2 != std::string::npos) cond = cond.substr(0, d2);
      // Signedness comes from the PTX type suffix, not the condition:
      // setp.gt.s32 %p, %r, -1 is a signed compare (vendor PTX from nvcc
      // 13 always emits .s32), so ignoring ".s" silently misjudges every
      // negative operand. Width follows the suffix; eq/ne are width-free.
      const bool is64 = (ty == "s64" || ty == "u64" || ty == "b64");
      const bool signed_cmp = (ty == "s32" || ty == "s64");
      int64_t xs = 0, ys = 0;
      if (is64) {
        xs = static_cast<int64_t>(xu);
        ys = static_cast<int64_t>(yu);
      } else {
        xs = static_cast<int32_t>(static_cast<uint32_t>(xu));
        ys = static_cast<int32_t>(static_cast<uint32_t>(yu));
      }
      bool r = false;
      if (cond == "eq") r = (is64 ? xu == yu : static_cast<uint32_t>(xu) == static_cast<uint32_t>(yu));
      else if (cond == "ne") r = (is64 ? xu != yu : static_cast<uint32_t>(xu) != static_cast<uint32_t>(yu));
      else if (signed_cmp) {
        if (cond == "ge") r = xs >= ys;
        else if (cond == "gt") r = xs > ys;
        else if (cond == "lt") r = xs < ys;
        else if (cond == "le") r = xs <= ys;
        else {
          if (error) *error = "unsupported setp condition '" + cond + "' in " + fn.name;
          return 0;
        }
      } else {
        const uint64_t x = is64 ? xu : static_cast<uint32_t>(xu);
        const uint64_t y = is64 ? yu : static_cast<uint32_t>(yu);
        if (cond == "ge") r = x >= y;
        else if (cond == "gt") r = x > y;
        else if (cond == "lt") r = x < y;
        else if (cond == "le") r = x <= y;
        else {
          if (error) *error = "unsupported setp condition '" + cond + "' in " + fn.name;
          return 0;
        }
      }
      st.preds[A[0]] = r;
      st.pc = next_pc;
      return 1;
    }

    // Memory fences and system-level barriers (membar.gl from __threadfence(),
    // membar.sys, fence.*). On a CPU backend every thread runs on its own
    // core over host memory, where the compiler and the CPU already provide
    // the release/acquire ordering these instructions request; the fence is
    // therefore a no-op that must never be emulated as a hardware counter.
    // Atomic RMWs that follow it are real host atomics (see the atomic block).
    if (op.rfind("membar.", 0) == 0 || op.rfind("fence.", 0) == 0) {
      std::atomic_thread_fence(std::memory_order_seq_cst);
      st.pc = next_pc;
      return 1;
    }

    // ld.global / ld.shared / ld.local / generic ld (no space qualifier):
    // the address resolves through mem_addr either way (registers hold real
    // host addresses in emulation, including the .local depot pointer).
    if ((op.rfind("ld.", 0) == 0 && op.rfind("ld.param.", 0) != 0)) {
      // ld.global.f32 %r, [%rd] (also ld.global.nc.*, ld.local.*, generic ld)
      if (A.size() < 2) { if (error) *error = "bad " + op + " in " + fn.name; return 0; }
      const uintptr_t addr = mem_addr(fn, st, A[1]);
      std::string pref;
      if (op.rfind("ld.global.nc.", 0) == 0) pref = "ld.global.nc.";
      else if (op.rfind("ld.global.", 0) == 0) pref = "ld.global.";
      else if (op.rfind("ld.shared.", 0) == 0) pref = "ld.shared.";
      else if (op.rfind("ld.local.", 0) == 0) pref = "ld.local.";
      else pref = "ld.";
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

    if (op.rfind("st.", 0) == 0) {
      // st.global.f32 [%rd], %r ; st.shared.f32 ; st.local.* ; generic st
      if (A.size() < 2) { if (error) *error = "bad " + op + " in " + fn.name; return 0; }
      const uintptr_t addr = mem_addr(fn, st, A[0]);
      std::string pref;
      if (op.rfind("st.global.", 0) == 0) pref = "st.global.";
      else if (op.rfind("st.shared.", 0) == 0) pref = "st.shared.";
      else if (op.rfind("st.local.", 0) == 0) pref = "st.local.";
      else pref = "st.";
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
      // The destination may carry a predicate: "shfl.sync.down.b32 %r10|%p1, ..."
      // Per PTX ISA 9.7.10.6, p is SET (not a guard): True if the computed
      // source lane is in range [0,31], False otherwise. The value is always
      // written to d.
      std::string dst = A[0];
      std::string setPred;
      {
        const size_t bar = dst.find('|');
        if (bar != std::string::npos) {
          setPred = dst.substr(bar + 1);
          dst = dst.substr(0, bar);
          if (!setPred.empty() && setPred[0] == '%') setPred = setPred.substr(1);
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
      // Bound model, hardware-verified on sm_75 with nvcc probes: the clamp
      // operand (c) is a RANGE BOUND, not a fallback lane. .down: valid iff
      // lane+off <= clamp; .up: valid iff lane>=off and lane-off >= clamp.
      // .bfly/.idx are always in range. Out-of-range lanes keep their own
      // value with p=False.
      const uint32_t clamp =
          A.size() >= 4 ? static_cast<uint32_t>(operand_u64(st, A[3])) & 0x1f
                        : 31;
      bool inRange = true;
      if (is_idx) {
        targetLane = sel & 0x1f;
        inRange = true;
      } else if (is_bfly) {
        targetLane = lane ^ (sel & 0x1f);
        inRange = true;
      } else if (is_up) {
        const uint32_t off = sel & 0x1f;
        if (lane < off || lane - off < clamp) {
          targetLane = lane;
          inRange = false;
        } else {
          targetLane = lane - off;
          inRange = true;
        }
      } else if (is_down) {
        const uint32_t off = sel & 0x1f;
        targetLane = lane + off;
        if (targetLane > clamp) {
          targetLane = lane;
          inRange = false;
        } else {
          inRange = true;
        }
      } else {
        targetLane = sel & 0x1f;
        inRange = true;
      }
      uint64_t val = st.reg(src);
      if (g_block_states) {
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
          inRange = false;
          val = st.reg(src);
        }
      }
      if (!inRange) val = st.reg(src);
      // Always write the value; set the predicate per PTX ISA.
      st.setreg(dst, val);
      if (!setPred.empty()) st.preds[setPred] = inRange;
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

    // ---- mma.sync.aligned (warp-cooperative matrix multiply-accumulate) ----
    // D = A*B + C over a logical MxNxK tile, gathered from the whole warp's
    // fragments (g_block_states, same mechanism as shfl/vote) and scattered
    // back to this thread's D fragment.
    //
    // Fragment layouts are PTX ISA 9.7.16.5 (m16n8k16 float/integer,
    // m16n8k8 float), NOT memory: groupID = lane>>2, tigid = lane&3, and the
    // per-element (row, col) formulas below are transcribed from those
    // sections verbatim.
    //
    // Accumulation order is UNSPECIFIED by the spec, so ours is documented
    // instead of guessed: k-ascending single-precision FMA chain per output
    // element (exact products for f16/bf16/fp8/u8/s8 inputs — all fit f32/s32
    // mantissas — rounded once per add, like hardware). The AOT path lowers
    // to the same order, so the two paths are bit-identical; hardware may
    // differ in the last ulp, which the spec explicitly permits.
    //
    // Supported: .row.col only (other layouts refuse loudly); shapes
    // m16n8k16 and m16n8k8; atype==btype in {f16,bf16,e4m3,e5m2,u8,s8};
    // dtype==ctype in {f32,f16,s32}. Everything else (sparse, wgmma, f64,
    // tf32, block-scaling, .col layouts) refuses loudly, never silently.
    if (op.rfind("mma.sync.", 0) == 0) {
      if (!g_block_states) {
        if (error) *error = "mma.sync outside a warp context in " + fn.name;
        return 0;
      }
      // Split "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32".
      std::vector<std::string> q;
      {
        std::stringstream ss(op);
        std::string t;
        while (std::getline(ss, t, '.')) q.push_back(t);
      }
      // q = [mma,sync,aligned?,shape,row,col,dtype,atype,btype,ctype]
      std::string shape, alayout, blayout, dtype, atype, btype, ctype;
      for (size_t i = 0; i < q.size(); ++i) {
        if (q[i] == "mma" || q[i] == "sync" || q[i] == "aligned") continue;
        if (q[i].rfind("m", 0) == 0 && q[i].find("n") != std::string::npos &&
            shape.empty())
          shape = q[i];
        else if ((q[i] == "row" || q[i] == "col") && alayout.empty())
          alayout = q[i];
        else if ((q[i] == "row" || q[i] == "col") && blayout.empty())
          blayout = q[i];
        else if (dtype.empty())
          dtype = q[i];
        else if (atype.empty()) atype = q[i];
        else if (btype.empty()) btype = q[i];
        else if (ctype.empty()) ctype = q[i];
      }
      const bool okShape = shape == "m16n8k16" || shape == "m16n8k8";
      const bool okLayout = alayout == "row" && blayout == "col";
      const bool okAB = (atype == "f16" || atype == "bf16" || atype == "e4m3" ||
                         atype == "e5m2" || atype == "u8" || atype == "s8") &&
                        atype == btype;
      const bool okDC = (dtype == "f32" || dtype == "f16" || dtype == "s32") &&
                        dtype == ctype;
      // Integer mma needs s32 accumulators; float mma needs float ones.
      const bool isInt = atype == "u8" || atype == "s8";
      const bool okMix = (!isInt && dtype != "s32") || (isInt && dtype == "s32");
      if (!okShape || !okLayout || !okAB || !okDC || !okMix) {
        if (error)
          *error = "unsupported mma form '" + op + "' in " + fn.name +
                   " (supported: .row.col m16n8k16/m16n8k8, "
                   "f16/bf16/e4m3/e5m2/u8/s8 -> f32/f16/s32)";
        return 0;
      }
      // K is the only shape dimension that varies the loop below (M=16, N=8
      // are structural in the tile arrays).
      const int K = (shape == "m16n8k16") ? 16 : 8;
      // Fragment arities per (shape, atype class).
      const bool fp16AB = atype == "f16" || atype == "bf16";
      int nA = 0, nB = 0;  // registers per fragment vector
      if (shape == "m16n8k16") {
        nA = fp16AB ? 4 : 2;
        nB = fp16AB ? 2 : 1;
      } else {
        nA = 2;
        nB = 1;
      }
      const int nD = (dtype == "f16") ? 2 : 4;
      // Split brace groups: {d..}, {a..}, {b..}, {c..}.
      std::vector<std::vector<std::string>> groups;
      {
        // Brace groups split operands: a token starting with '{' opens a new
        // group (so a lone "{%r9}" is its own group instead of gluing to the
        // address token before it); a token ending with '}' closes it.
        std::vector<std::string> cur;
        for (const auto& a : A) {
          const std::string t = trim(a);
          if (!t.empty() && t.front() == '{' && !cur.empty()) {
            groups.push_back(cur);
            cur.clear();
          }
          cur.push_back(vector_reg_name(a));
          if (!a.empty() && a.back() == '}') {
            groups.push_back(cur);
            cur.clear();
          }
        }
        if (!cur.empty()) groups.push_back(cur);
      }
      if (groups.size() != 4 || (int)groups[0].size() != nD ||
          (int)groups[1].size() != nA || (int)groups[2].size() != nB ||
          (int)groups[3].size() != nD) {
        if (error)
          *error = "bad mma fragment groups in '" + op + "' in " + fn.name;
        return 0;
      }
      // Element fetchers: register value -> float (or int64 for the integer
      // path). f16x2/bf16x2 regs hold two elements (low first); .b32 regs
      // hold four bytes (low first); f32/s32 regs hold one. The `f16` flag
      // selects the element format explicitly: A/B fragments use atype,
      // C/D accumulators are always plain .f16 when dtype is f16.
      const auto getF16x2 = [&](const ExecState& s, const std::string& r, int idx,
                                bool isBf16) -> float {
        const uint32_t v = static_cast<uint32_t>(s.reg(r));
        const uint16_t h = static_cast<uint16_t>((v >> (16 * (idx & 1))) & 0xffffu);
        return isBf16 ? bf16_to_f32(h) : f16_to_f32(h);
      };
      const auto getB32 = [&](const ExecState& s, const std::string& r,
                              int idx) -> float {
        const uint32_t v = static_cast<uint32_t>(s.reg(r));
        const uint8_t byte = static_cast<uint8_t>((v >> (8 * idx)) & 0xffu);
        if (atype == "u8") return static_cast<float>(byte);
        if (atype == "s8") return static_cast<float>(static_cast<int8_t>(byte));
        const NarrowFpFormat& F = (atype == "e4m3") ? kE4M3 : kE5M2;
        return narrow_to_f32(byte, F);
      };
      const auto getI32 = [&](const ExecState& s, const std::string& r) -> int64_t {
        return static_cast<int64_t>(static_cast<int32_t>(s.reg(r)));
      };
      // Full logical tiles, gathered across the warp.
      float tA[16][16], tB[16][8], tC[16][8];
      int64_t iA[16][16], iB[16][8], iC[16][8];
      for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 16; ++j) {
          tA[i][j] = 0.0f;
          iA[i][j] = 0;
        }
      for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 8; ++j) {
          tB[i][j] = 0.0f;
          tC[i][j] = 0.0f;
          iB[i][j] = 0;
          iC[i][j] = 0;
        }
      const uint32_t linear0 =
          st.tid[0] + st.tid[1] * st.ntid[0] + st.tid[2] * st.ntid[0] * st.ntid[1];
      const uint32_t warpBase0 = linear0 - (linear0 % 32);
      for (uint32_t l = 0; l < 32; ++l) {
        const size_t si = static_cast<size_t>(warpBase0 + l);
        if (si >= g_block_states->size()) continue;
        const ExecState& ls = (*g_block_states)[si];
        const int g = static_cast<int>(l >> 2);
        const int t = static_cast<int>(l & 3);
        // --- A fragment ---
        if (shape == "m16n8k16" && fp16AB) {
          for (int s = 0; s < 8; ++s) {
            const std::string& reg = groups[1][s / 2];
            const float v = getF16x2(ls, reg, s % 2, atype == "bf16");
            int r, c;
            if (s < 2 || (s >= 4 && s < 6)) r = g;
            else r = g + 8;
            c = (s < 4) ? (t * 2 + (s & 1)) : (t * 2 + (s & 1) + 8);
            tA[r][c] = v;
          }
        } else if (shape == "m16n8k16") {
          // e4m3/e5m2/u8/s8: 2x .b32, a0..a7.
          for (int s = 0; s < 8; ++s) {
            const std::string& reg = groups[1][s / 4];
            const int r = (s < 4) ? g : g + 8;
            const int c = t * 4 + (s & 3);
            if (isInt) {
              const uint32_t v = static_cast<uint32_t>(ls.reg(reg));
              const uint8_t byte = static_cast<uint8_t>((v >> (8 * (s & 3))) & 0xffu);
              iA[r][c] = (atype == "s8") ? static_cast<int64_t>(static_cast<int8_t>(byte))
                                         : static_cast<int64_t>(byte);
            } else {
              tA[r][c] = getB32(ls, reg, s & 3);
            }
          }
        } else {
          // m16n8k8 f16/bf16: 2x .f16x2, a0..a3.
          for (int s = 0; s < 4; ++s) {
            const std::string& reg = groups[1][s / 2];
            const float v = getF16x2(ls, reg, s % 2, atype == "bf16");
            const int r = (s < 2) ? g : g + 8;
            const int c = t * 2 + (s & 1);
            tA[r][c] = v;
          }
        }
        // --- B fragment ---
        if (fp16AB) {
          const int nb = (shape == "m16n8k16") ? 4 : 2;
          for (int s = 0; s < nb; ++s) {
            const std::string& reg = groups[2][s / 2];
            const float v = getF16x2(ls, reg, s % 2, atype == "bf16");
            int r, c;
            if (shape == "m16n8k16") {
              r = (s < 2) ? (t * 2 + (s & 1)) : (t * 2 + (s & 1) + 8);
              c = g;
            } else {
              r = t * 2 + s;
              c = g;
            }
            tB[r][c] = v;
          }
        } else {
          for (int s = 0; s < 4; ++s) {
            const std::string& reg = groups[2][0];
            const int r = t * 4 + s;
            const int c = g;
            if (isInt) {
              const uint32_t v = static_cast<uint32_t>(ls.reg(reg));
              const uint8_t byte = static_cast<uint8_t>((v >> (8 * s)) & 0xffu);
              iB[r][c] = (btype == "s8") ? static_cast<int64_t>(static_cast<int8_t>(byte))
                                         : static_cast<int64_t>(byte);
            } else {
              tB[r][c] = getB32(ls, reg, s);
            }
          }
        }
        // --- C fragment (f32: 4 regs; f16: 2x .f16x2; s32: 4 regs) ---
        if (dtype == "f16") {
          for (int s = 0; s < 4; ++s) {
            const std::string& reg = groups[3][s / 2];
            const float v = getF16x2(ls, reg, s % 2, false);
            const int r = (s < 2) ? g : g + 8;
            const int c = t * 2 + (s & 1);
            tC[r][c] = v;
          }
        } else if (isInt) {
          for (int s = 0; s < 4; ++s) {
            const std::string& reg = groups[3][s];
            const int r = (s < 2) ? g : g + 8;
            const int c = t * 2 + (s & 1);
            iC[r][c] = getI32(ls, reg);
          }
        } else {
          for (int s = 0; s < 4; ++s) {
            const std::string& reg = groups[3][s];
            const uint32_t v = static_cast<uint32_t>(ls.reg(reg));
            const int r = (s < 2) ? g : g + 8;
            const int c = t * 2 + (s & 1);
            tC[r][c] = bit_cast_v<float>(v);
          }
        }
      }
      // --- compute + scatter this thread's D fragment ---
      const uint32_t lane =
          linear0 - warpBase0;
      const int g0 = static_cast<int>(lane >> 2);
      const int t0 = static_cast<int>(lane & 3);
      if (isInt) {
        int64_t acc[4];
        for (int s = 0; s < 4; ++s) {
          const int r = (s < 2) ? g0 : g0 + 8;
          const int c = t0 * 2 + (s & 1);
          int64_t sum = iC[r][c];
          for (int k = 0; k < K; ++k) sum += iA[r][k] * iB[k][c];
          acc[s] = sum;
        }
        for (int s = 0; s < 4; ++s)
          st.setreg(groups[0][s], static_cast<uint64_t>(static_cast<uint32_t>(acc[s])));
      } else {
        float acc[4];
        for (int s = 0; s < 4; ++s) {
          const int r = (s < 2) ? g0 : g0 + 8;
          const int c = t0 * 2 + (s & 1);
          float sum = tC[r][c];
          for (int k = 0; k < K; ++k) sum = std::fma(tA[r][k], tB[k][c], sum);
          acc[s] = sum;
        }
        if (dtype == "f16") {
          for (int s = 0; s < 2; ++s) {
            const uint16_t lo = f32_to_f16_rne(acc[s * 2]);
            const uint16_t hi = f32_to_f16_rne(acc[s * 2 + 1]);
            st.setreg(groups[0][s],
                      static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 16));
          }
        } else {
          for (int s = 0; s < 4; ++s)
            st.setreg(groups[0][s], static_cast<uint64_t>(bit_cast_v<uint32_t>(acc[s])));
        }
      }
      st.pc = next_pc;
      return 1;
    }

    // ---- ldmatrix / stmatrix (warp-collective shared-memory matrix move) ----
    // One or more 8x8 .b16 matrices collectively moved between shared memory
    // and registers, PTX ISA 9.7.16.5.15/16 (only .m8n8/.b16: the sm_75 form
    // real kernels use; .m16n16/.m8n16/.b8/.s4 are sm_90+/sm_100a+ and refuse
    // loudly below).
    //
    // Addressing (transcribed from the spec): eight threads per matrix provide
    // row-start addresses (threads 0-7 matrix 0, 8-15 matrix 1, 16-23 matrix 2,
    // 24-31 matrix 3); a group of four consecutive threads moves a whole row
    // and EVERY thread receives a fragment of EVERY matrix: thread t takes row
    // t%8, elements (t%4)*2 and +1. .trans swaps the receive side to columns
    // (the addresses still name rows, so coverage stays complete).
    if (op.rfind("ldmatrix.sync.", 0) == 0 || op.rfind("stmatrix.sync.", 0) == 0) {
      const bool isLoad = op[0] == 'l';
      if (!g_block_states) {
        if (error) *error = std::string(isLoad ? "ldmatrix" : "stmatrix") +
                            " outside a warp context in " + fn.name;
        return 0;
      }
      std::vector<std::string> q;
      {
        std::stringstream ss(op);
        std::string t;
        while (std::getline(ss, t, '.')) q.push_back(t);
      }
      // q = [ldmatrix,sync,aligned,m8n8,x4,trans?,shared(::cta)?,b16]
      std::string shape, num, type;
      bool trans = false, sharedQ = false;
      for (const auto& t : q) {
        if (t == "ldmatrix" || t == "stmatrix" || t == "sync" || t == "aligned")
          continue;
        else if (t == "m8n8") shape = t;
        else if (t == "x1" || t == "x2" || t == "x4") num = t;
        else if (t == "trans") trans = true;
        else if (t == "shared" || t == "shared::cta") sharedQ = true;
        else if (t == "b16" || t == "b8") type = t;
        else {
          if (error)
            *error = "unsupported " + std::string(isLoad ? "ldmatrix" : "stmatrix") +
                     " qualifier '" + t + "' in " + fn.name;
          return 0;
        }
      }
      (void)sharedQ;  // generic addresses into shared resolve the same way
      if (shape != "m8n8" || type != "b16" ||
          (num != "x1" && num != "x2" && num != "x4")) {
        if (error)
          *error = "unsupported " + std::string(isLoad ? "ldmatrix" : "stmatrix") +
                   " form '" + op + "' in " + fn.name +
                   " (supported: .m8n8.x{1,2,4}[.trans].b16)";
        return 0;
      }
      const int nMat = (num == "x4") ? 4 : (num == "x2") ? 2 : 1;
      // Operands: ldmatrix {d..}, [addr] | stmatrix [addr], {r..}.
      std::vector<std::vector<std::string>> groups;
      {
        // Same brace-group rule as ldmatrix above (a lone "{b0}" is its own
        // group, e.g. the single-.b32 B fragment of integer/fp8 mma).
        std::vector<std::string> cur;
        for (const auto& a : A) {
          const std::string t = trim(a);
          if (!t.empty() && t.front() == '{' && !cur.empty()) {
            groups.push_back(cur);
            cur.clear();
          }
          cur.push_back(vector_reg_name(a));
          if (!a.empty() && a.back() == '}') {
            groups.push_back(cur);
            cur.clear();
          }
        }
        if (!cur.empty()) groups.push_back(cur);
      }
      // After grouping: ldmatrix -> [{d..}, {addr}]; stmatrix -> [{addr}, {r..}].
      // The address token keeps its brackets ([addr] / [reg+off]).
      if (groups.size() != 2 || (int)groups[isLoad ? 0 : 1].size() != nMat) {
        if (error)
          *error = "bad " + std::string(isLoad ? "ldmatrix" : "stmatrix") +
                   " operands in '" + op + "' in " + fn.name;
        return 0;
      }
      const std::vector<std::string>& fragRegs = groups[isLoad ? 0 : 1];
      const std::string addrTok = groups[isLoad ? 1 : 0][0];
      // Address operand: [reg] or [reg+off] (spec: Addresses as Operands).
      const std::string inner = strip_brackets(addrTok);
      std::string addrBase;
      int64_t addrOff = 0;
      {
        std::string base;
        int64_t off = 0;
        if (split_reg_offset(inner, &base, &off)) {
          addrBase = base;
          addrOff = off;
        } else {
          addrBase = inner;
        }
      }
      const uint32_t linear0 =
          st.tid[0] + st.tid[1] * st.ntid[0] + st.tid[2] * st.ntid[0] * st.ntid[1];
      const uint32_t warpBase0 = linear0 - (linear0 % 32);
      // Row-start address per lane (threads that provide none still carry a
      // usable address in well-formed code; sm_75 note).
      uintptr_t rowAddr[32];
      for (uint32_t l = 0; l < 32; ++l) {
        const size_t si = static_cast<size_t>(warpBase0 + l);
        if (si >= g_block_states->size()) {
          rowAddr[l] = 0;
          continue;
        }
        const ExecState& ls = (*g_block_states)[si];
        uint64_t base = 0;
        // The address operand names a register (possibly with +/- offset).
        // A bare .shared symbol is also legal PTX ([smem]); resolve it.
        {
          uintptr_t sym = 0;
          if (shared_symbol_addr(fn, ls, addrBase, &sym))
            base = sym;
          else if (local_symbol_addr(fn, ls, addrBase, &sym))
            base = sym;
          else
            base = ls.reg(addrBase);
        }
        rowAddr[l] = static_cast<uintptr_t>(base + static_cast<uint64_t>(addrOff));
      }
      // Every thread receives a fragment from EVERY matrix (PTX ISA
      // 9.7.16.5.15: "each component of the vector expression holds a fragment
      // from the corresponding matrix"; for .x2 the second matrix lands in the
      // next register *in each thread*). Thread t takes row t>>2, elements
      // (t%4)*2 and +1: a group of four consecutive threads moves one whole
      // row (16 bytes), thread 0 the first fragment, and so on. The row's
      // start address comes from the lane that provided it: threads 0-7 for
      // matrix 0, 8-15 for matrix 1, and so on (the .num table in the spec).
      // With .trans the thread takes column t>>2, rows (t%4)*2 instead; the
      // addresses still name rows, so coverage holds. This is the layout that
      // feeds mma.sync directly (ldmatrix fragments ARE mma fragments).
      for (uint32_t l = 0; l < 32; ++l) {
        const size_t si = static_cast<size_t>(warpBase0 + l);
        if (si >= g_block_states->size()) continue;
        const int e0 = static_cast<int>((l % 4) * 2);
        const int fixed = static_cast<int>(l >> 2);
        for (int m = 0; m < nMat; ++m) {
          uint16_t els[2] = {0, 0};
          for (int e = 0; e < 2; ++e) {
            const int r = trans ? (e0 + e) : fixed;
            const int c = trans ? fixed : (e0 + e);
            // Lane that supplied matrix m's row r's address.
            const uint32_t provider = static_cast<uint32_t>(m * 8 + r);
            if (provider >= 32) continue;
            const uintptr_t at =
                rowAddr[provider] + static_cast<uintptr_t>(c * 2);
            if (isLoad) {
              uint16_t v = 0;
              std::memcpy(&v, reinterpret_cast<const void*>(at), 2);
              els[e] = v;
            } else {
              // stmatrix: this thread's fragment register for matrix m goes to
              // the shared window at the same (row, column) position.
              const uint32_t v = static_cast<uint32_t>(
                  (*g_block_states)[si].reg(fragRegs[m]));
              const uint16_t bits = static_cast<uint16_t>(
                  (e == 0 ? (v & 0xffffu) : ((v >> 16) & 0xffffu)));
              std::memcpy(reinterpret_cast<void*>(at), &bits, 2);
            }
          }
          if (isLoad) {
            (*g_block_states)[si].setreg(
                fragRegs[m], static_cast<uint64_t>(els[0]) |
                                (static_cast<uint64_t>(els[1]) << 16));
          }
        }
      }
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

    // Block-wide barrier (bar.sync, its new spelling barrier.sync, and the
    // warp-scoped bar.warp.sync): the CTA scheduler keeps the thread parked
    // until all live threads of the block reach this point (see
    // execute_kernel).
    if (op == "bar.sync" || op == "bar.warp.sync" || op == "barrier.sync") {
      st.pc = next_pc;
      return 2;
    }

    // Метка ("NAME:", в т.ч. "$L__BB0_2:" от nvcc 13), проскочившая
    // мимо фронтенда — no-op, а не неподдерживаемая инструкция.
    // Опкод в PTX никогда не содержит ':', поэтому op с хвостом ':'
    // однозначно метка; имя проверяется по правилам идентификатора.
    if (!op.empty() && op.back() == ':') {
      const std::string name = op.substr(0, op.size() - 1);
      bool ok = !name.empty() && ident_start(name[0]);
      for (char c : name)
        if (!ident_cont(c)) { ok = false; break; }
      if (ok) { st.pc = next_pc; return 1; }
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
                    std::string* error,
                    const std::unordered_map<std::string, uint64_t>* globalBases) {
  if (fnIndex < 0 || fnIndex >= static_cast<int>(prog.functions.size())) {
    if (error) *error = "kernel function index out of range";
    return false;
  }
  const PtxFunction& fn = prog.functions[fnIndex];

  // Dynamic .shared (unsized ".extern .shared") has no launch-time backing in
  // the interpreter: refusing loudly beats a scratchpad overrun.
  for (const auto& sv : fn.sharedVars)
    if (sv.dynamic) {
      if (error)
        *error = "kernel '" + fn.name + "' uses dynamic .shared '" + sv.name +
                 "' (extern, unsized): launch-time size is not supported by " +
                 "the interpreter";
      return false;
    }

  // CTA-sized thread states for one block, stepped in lockstep (SIMT).
  const size_t threadsPerBlock = static_cast<size_t>(bz) * by * bx;

  // Per-block .shared scratchpad, one mapping reused by every block (blocks
  // are stepped sequentially, so a single region suffices).
  //
  // The region is mapped LOW (MAP_32BIT) on purpose: nvcc addresses shared
  // arrays with 32-bit arithmetic ("mov.u32 %r, sym; add.s32 %r, %r, %off"),
  // and a scratchpad above 4GB would lose its high bits in that roundtrip and
  // fault on st.shared — the exact reason the AOT runtime maps its scratchpad
  // the same way (rouge_runtime.cpp: rouge_smem_alloc).
  size_t smemBytes = fn.sharedSize > 0 ? static_cast<size_t>(fn.sharedSize) : 1;
  void* smemMap = nullptr;
  std::vector<uint8_t> smemHeap;
#if defined(__linux__) && defined(__x86_64__)
  const size_t need = smemBytes + 64;
  void* map = mmap(nullptr, need, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_32BIT, -1, 0);
  if (map != MAP_FAILED) smemMap = map;
#endif
  if (smemMap == nullptr) {
    smemHeap.assign(smemBytes, 0);  // fallback: heap (offsets then must fit 32 bits)
    smemMap = smemHeap.data();
  }
  auto smem_release = [&]() {
#if defined(__linux__) && defined(__x86_64__)
    if (!smemHeap.empty()) munmap(smemMap, smemBytes + 64);
#endif
  };

  // Module-global backing storage (if the API layer provided any) is
  // visible to every address-resolution path for the whole launch.
  g_global_bases = globalBases;

  for (uint64_t bz_i = 0; bz_i < gz; ++bz_i)
    for (uint64_t by_i = 0; by_i < gy; ++by_i)
      for (uint64_t bx_i = 0; bx_i < gx; ++bx_i) {
        // Per-block scratchpad for .shared variables (one per CTA): the low
        // mapping is zeroed fresh for every block, since each CTA sees its own
        // copy of shared memory.
        std::memset(smemMap, 0, smemBytes);
        const uintptr_t sharedBase = reinterpret_cast<uintptr_t>(smemMap);

        std::vector<ExecState> states(threadsPerBlock);
        std::vector<bool> alive(threadsPerBlock, true);
        std::vector<bool> blocked(threadsPerBlock, false);
        size_t n = 0;
        for (uint64_t tz = 0; tz < bz; ++tz)
          for (uint64_t ty = 0; ty < by; ++ty)
            for (uint64_t tx = 0; tx < bx; ++tx) {
              ExecState& st = states[n++];
              st.sharedBase = sharedBase;
              if (fn.localSize > 0) {
                st.local.assign(static_cast<size_t>(fn.localSize), 0);
                st.localBase = reinterpret_cast<uintptr_t>(st.local.data());
              }
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
              if (error)
                *error = "emulation step limit exceeded (runaway loop?) in " + fn.name;
              g_block_states = nullptr;
              g_global_bases = nullptr;
              smem_release();
              return false;
            }
            const int rc = run_thread_step(fn, paramsBlob, states[i], error);
            if (rc == 0) {
              g_block_states = nullptr;
              g_global_bases = nullptr;
              smem_release();
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
  g_global_bases = nullptr;
  smem_release();
  return true;
}

}  // namespace rouge