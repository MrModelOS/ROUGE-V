// rungpu.c -- CUDA Driver API test stand for the ROUGE-V translated kernels.
//
// Loads a PTX file produced by
//     ptx2ir --target nvptx64-nvidia-cuda <k>.ptx /tmp/<k>.ll
//     clang --target=nvptx64-nvidia-cuda -march=sm_75 -S /tmp/<k>.ll -o <k>.ptx
// through cuModuleLoadData (driver-side JIT), launches the requested kernel on
// the real card and checks the result against a host reference computed from
// the ORIGINAL hand-written PTX (software/rouge-compiler/tests/kernels/*.ptx).
//
// The f16/bf16 reference is bit-exact: every f16 and bf16 operation of the
// source PTX is emulated on the host with correct round-to-nearest-even at
// binary16 / bfloat16 precision.  No tolerance is used anywhere; every
// comparison is on raw bit patterns.
//
// usage:  rungpu <file.ptx> <vadd|block_reduce|fp16_reduce|gemm_tile>
//         rungpu <file.ptx> all
//
// exit:   0 = kernel verified, 1 = result mismatch, 2 = API error,
//         3 = device fault while the kernel was running.
//
// Build:  gcc rungpu.c -o rungpu -I/opt/cuda/targets/x86_64-linux/include
//         -L/usr/lib -l:libcuda.so.1 -lm

#include <cuda.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// ---------------------------------------------------------------------------
// CUDA error plumbing
// ---------------------------------------------------------------------------
static int g_exit_code = 0;

static const char* cuerr(CUresult r) {
  const char* n = "?";
  cuGetErrorName(r, &n);
  return n;
}
static const char* cuerrstr(CUresult r) {
  const char* s = "?";
  cuGetErrorString(r, &s);
  return s;
}
static void ck(CUresult r, const char* what, int line) {
  if (r != CUDA_SUCCESS) {
    printf("  FATAL: %s -> %s (%s) at line %d\n", what, cuerr(r), cuerrstr(r), line);
    fflush(stdout);
    exit(2);
  }
}

// Create a context, retrying once: the driver occasionally answers the first
// cuCtxCreate after a burst of launches with a transient CUDA_ERROR_UNKNOWN.
// One bounded retry with a pause distinguishes that from a real failure; the
// retry is always reported, never silent.
static int make_ctx(CUcontext* ctx, CUdevice dev) {
  CUresult r = cuCtxCreate(ctx, 0, dev, 0);
  if (r != CUDA_SUCCESS) {
    const char* n_;
    cuGetErrorName(r, &n_);
    printf("  (cuCtxCreate retry after %s, pausing 5 s)\n", n_);
    CUresult r2 = cuCtxSynchronize();
    (void)r2;
    struct timespec ts;
    ts.tv_sec = 5;
    ts.tv_nsec = 0;
    nanosleep(&ts, 0);
    r = cuCtxCreate(ctx, 0, dev, 0);
  }
  if (r != CUDA_SUCCESS) {
    const char* n_;
    cuGetErrorName(r, &n_);
    printf("FAIL cuCtxCreate: %s\n", n_);
    return 1;
  }
  return 0;
}

#define CK(x) ck((x), #x, __LINE__)

// ---------------------------------------------------------------------------
// bit-level f16 / bf16 primitives
// ---------------------------------------------------------------------------

// Round a finite long double to binary16, round-to-nearest-even.
// Also accepts +/-inf / NaN (routed through the f32 path, which is lossless).
static uint16_t ld_to_f16_bits(long double v) {
  if (!isfinite((double)v) && !isinf((float)v) && !isnan((float)v)) {
    /* long double value beyond float range: the only finite values that
       matter here are far outside binary16, so they saturate. */
    return (v < 0) ? 0xfc00u : 0x7c00u;
  }
  if (isinf((float)v) || isnan((float)v)) {
    float f = (float)v;
    uint32_t u;
    memcpy(&u, &f, 4);
    uint16_t sign = (uint16_t)((u >> 16) & 0x8000u);
    if ((u & 0x7fffffffu) > 0x7f800000u) { /* NaN */
      uint16_t frac = (uint16_t)((u >> 13) & 0x3ffu);
      return (uint16_t)(sign | 0x7c00u | (frac ? frac : 1u));
    }
    return (uint16_t)(sign | 0x7c00u);
  }
  if (v == 0.0L) return signbit(v) ? 0x8000u : 0x0000u;

  int s = signbit(v) ? 1 : 0;
  long double av = fabsl(v);
  int k = ilogbl(av); /* floor(log2(av)) */
  if (k < -14) {     /* binary16 subnormal: quantum is 2^-24 */
    long double q = ldexpl(av, 24); /* in [0, 1024) */
    uint64_t n = (uint64_t)q;
    long double fr = q - (long double)n;
    if (fr > 0.5L || (fr == 0.5L && (n & 1u))) n++;
    if (n >= 1024u) return (uint16_t)((s << 15) | 0x0400u); /* rounded up to min normal */
    return (uint16_t)((s << 15) | (uint16_t)n);
  }
  /* normal: 11 significant bits */
  long double q = ldexpl(av, 10 - k); /* in [1024, 2048) */
  uint64_t n = (uint64_t)q;
  long double fr = q - (long double)n;
  if (fr > 0.5L || (fr == 0.5L && (n & 1u))) n++;
  if (n >= 2048u) {
    n = 1024u;
    k++;
  }
  if (k > 15) return (uint16_t)((s << 15) | 0x7c00u); /* overflow -> inf */
  return (uint16_t)((s << 15) | (uint16_t)((k + 15) * 1024 + (int)(n - 1024)));
}

static uint16_t f32_to_f16_bits(float f) { return ld_to_f16_bits((long double)f); }

static float f16_bits_to_f32(uint16_t b) {
  uint32_t sign = (uint32_t)(b & 0x8000u) << 16;
  uint32_t exp = (b >> 10) & 0x1fu;
  uint32_t man = b & 0x3ffu;
  uint32_t u;
  if (exp == 0) {
    if (man == 0) {
      u = sign;
    } else { /* subnormal: normalise */
      int e = -1;
      uint32_t m = man;
      do {
        m <<= 1;
        e++;
      } while (!(m & 0x400u));
      u = sign | ((uint32_t)(127 - 15 - e) << 23) | ((m & 0x3ffu) << 13);
    }
  } else if (exp == 0x1f) {
    u = sign | 0x7f800000u | (man << 13);
  } else {
    u = sign | ((exp + 127 - 15) << 23) | (man << 13);
  }
  float f;
  memcpy(&f, &u, 4);
  return f;
}

// round-to-nearest-even f32 -> bf16, matching the ROUGE-V lowering
//   bfe u32 %r, %f, 16, 1 ; add %r, %r, %f ; add %r, %r, 32767 ; shr %r, %r, 16
static uint16_t f32_to_bf16_bits(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) { /* NaN */
    return (uint16_t)((u >> 16) | 0x0040u);
  }
  uint32_t lsb = (u >> 16) & 1u;
  u += 0x7fffu + lsb;
  return (uint16_t)(u >> 16);
}
static float bf16_bits_to_f32(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

// ---- f16 arithmetic with a single rounding, at binary16 precision ----------
static uint16_t f16_add(uint16_t a, uint16_t b) {
  return f32_to_f16_bits(f16_bits_to_f32(a) + f16_bits_to_f32(b));
}
static uint16_t f16_mul(uint16_t a, uint16_t b) {
  return f32_to_f16_bits(f16_bits_to_f32(a) * f16_bits_to_f32(b));
}
// fma.rn.f16: one single rounding of the *exact* a*b+c.
// 80-bit long double is wide enough: the exact product of two binary16 values
// needs <= 22 significand bits, and any c whose exponent is more than 42 away
// cannot change the binary16 rounding.
static uint16_t f16_fma(uint16_t a, uint16_t b, uint16_t c) {
  long double ex = (long double)f16_bits_to_f32(a) * (long double)f16_bits_to_f32(b) +
                   (long double)f16_bits_to_f32(c);
  return ld_to_f16_bits(ex);
}
// what the *translated* kernel does: fma.rn.f32 then a rounding to f16
static uint16_t f16_fma_double_rounded(uint16_t a, uint16_t b, uint16_t c) {
  return f32_to_f16_bits(fmaf(f16_bits_to_f32(a), f16_bits_to_f32(b), f16_bits_to_f32(c)));
}
static uint16_t bf16_add(uint16_t a, uint16_t b) {
  return f32_to_bf16_bits(bf16_bits_to_f32(a) + bf16_bits_to_f32(b));
}

// ---- cross-check the hand-rolled f16 model against the compiler's _Float16 -
static uint32_t g_model_mismatch = 0;
static uint32_t g_model_checked = 0;

static void model_crosscheck(void) {
  static uint64_t s = 0x243f6a8885a308d3ull;
  g_model_mismatch = 0;
  g_model_checked = 0;
  /* (a) every f16 bit pattern: f16 -> f32 -> f16 must be the identity.
     Exception: the 1022 NaN patterns -- widening a signalling NaN to f32 and
     narrowing it back legitimately quiets the NaN, so those are skipped. */
  for (uint32_t b = 0; b < 65536u; ++b) {
    uint16_t p = (uint16_t)b, q = f32_to_f16_bits(f16_bits_to_f32(p));
    g_model_checked++;
    if (p == q) continue;
    if (((p >> 10) & 0x1fu) == 0x1fu && (p & 0x3ffu)) continue; /* NaN */
    if (g_model_mismatch < 4)
      printf("  [model] f16 roundtrip broke: 0x%04x -> 0x%04x\n", p, q);
    g_model_mismatch++;
  }
  /* (b) random f32 bit patterns: manual RNE vs. the compiler's _Float16 */
  for (long i = 0; i < 3000000; ++i) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    uint32_t u = (uint32_t)(s >> 32);
    /* only finite, in the range where f16 is not trivially inf/zero */
    u = (u & 0x807fffffu) | ((uint32_t)(118u + (s & 7u)) << 23);
    float f;
    memcpy(&f, &u, 4);
    _Float16 h = (_Float16)f;
    uint16_t got;
    memcpy(&got, &h, 2);
    uint16_t mine = f32_to_f16_bits(f);
    if (got != mine) {
      if (g_model_mismatch < 8)
        printf("  [model] f32->f16 %.9g : _Float16=0x%04x manual=0x%04x\n", (double)f, got, mine);
      g_model_mismatch++;
    }
    g_model_checked++;
  }
  /* (c) f16 add / mul: manual vs. the compiler's _Float16 operators */
  for (long i = 0; i < 2000000; ++i) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    uint16_t a = (uint16_t)(s >> 16), b = (uint16_t)(s >> 32);
    a = (uint16_t)((a & 0x83ffu) | 0x2800u); /* keep |value| sane */
    b = (uint16_t)((b & 0x83ffu) | 0x2800u);
    _Float16 ha, hb;
    memcpy(&ha, &a, 2);
    memcpy(&hb, &b, 2);
    uint16_t got, mine;
    _Float16 hr = ha + hb;
    memcpy(&got, &hr, 2);
    mine = f16_add(a, b);
    if (got != mine) g_model_mismatch++;
    hr = ha * hb;
    memcpy(&got, &hr, 2);
    mine = f16_mul(a, b);
    if (got != mine) g_model_mismatch++;
    g_model_checked += 2;
  }
}

// ---------------------------------------------------------------------------
// deterministic PRNG
// ---------------------------------------------------------------------------
static uint64_t g_rng = 0x9e3779b97f4a7c15ull;
static void rng_seed(uint64_t s) { g_rng = s ? s : 1; }
static uint32_t rnd32(void) {
  g_rng ^= g_rng << 13;
  g_rng ^= g_rng >> 7;
  g_rng ^= g_rng << 17;
  return (uint32_t)(g_rng >> 32);
}
static float rnd_f32_in(float lo, float hi) {
  return lo + (hi - lo) * ((float)(rnd32() & 0xffffff) / 16777216.0f);
}
/* random f16 that is exactly representable, value in [-2, 2) */
static uint16_t rnd_f16_small(void) {
  int32_t m = (int32_t)(rnd32() % 2048u) - 1024; /* -1024..1023 */
  return f32_to_f16_bits((float)m / 512.0f);
}

// ---------------------------------------------------------------------------
// generic report helper
// ---------------------------------------------------------------------------
static void dump_words(const char* tag, const void* p, size_t n, int as16) {
  printf("    %-22s", tag);
  for (size_t i = 0; i < n && i < 8; ++i) {
    if (as16)
      printf(" 0x%04x", ((const uint16_t*)p)[i]);
    else
      printf(" 0x%08x", ((const uint32_t*)p)[i]);
  }
  printf("\n");
}

#define SENTINEL_F32 0xdeadbeefu

// ---------------------------------------------------------------------------
// kernel: vadd -- c[i] = a[i] + b[i], flattened index i < n
// ---------------------------------------------------------------------------
static int test_vadd(CUmodule mod) {
  const int N = 1024, threads = 256, blocks = (N + threads - 1) / threads;
  printf("  config: N=%d blockDim=(%d,1,1) gridDim=(%d,1,1)\n", N, threads, blocks);

  CUfunction fn;
  CK(cuModuleGetFunction(&fn, mod, "vadd"));

  float* ha = malloc(N * 4);
  float* hb = malloc(N * 4);
  float* hc = malloc(N * 4);
  float* he = malloc(N * 4);
  for (int i = 0; i < N; ++i) {
    ha[i] = (float)i * 0.25f;
    hb[i] = (float)(i * 2) - 0.5f;
    he[i] = ha[i] + hb[i];
  }
  memset(hc, 0x5a, N * 4);

  CUdeviceptr da, db, dc;
  CK(cuMemAlloc(&da, N * 4));
  CK(cuMemAlloc(&db, N * 4));
  CK(cuMemAlloc(&dc, N * 4));
  CK(cuMemcpyHtoD(da, ha, N * 4));
  CK(cuMemcpyHtoD(db, hb, N * 4));
  CK(cuMemcpyHtoD(dc, hc, N * 4));

  int n = N;
  void* args[] = {&da, &db, &dc, &n};
  CK(cuLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, 0, args, 0));
  CUresult r = cuCtxSynchronize();
  if (r != CUDA_SUCCESS) {
    printf("  MISMATCH: launch/sync failed: %s (%s)\n", cuerr(r), cuerrstr(r));
    return 3;
  }
  CK(cuMemcpyDtoH(hc, dc, N * 4));

  int bad = 0, first = -1;
  for (int i = 0; i < N; ++i)
    if (memcmp(&hc[i], &he[i], 4) != 0) {
      if (first < 0) first = i;
      bad++;
    }
  if (!bad) {
    printf("  OK: %d elements, c[i] = a[i] + b[i] verified bit-exactly\n", N);
    return 0;
  }
  printf("  MISMATCH: %d of %d elements wrong, first at i=%d\n", bad, N, first);
  dump_words("got", hc + first, 4, 0);
  dump_words("expected", he + first, 4, 0);
  return 1;
}

// ---------------------------------------------------------------------------
// kernel: block_reduce -- out[ctaid.x] = sum(data[ctaid.x*ntid.x .. +ntid.x))
// ---------------------------------------------------------------------------
static int test_block_reduce(CUmodule mod) {
  const int threads = 256, blocks = 8, N = threads * blocks;
  printf("  config: N=%d blockDim=(%d,1,1) gridDim=(%d,1,1), out[ctaid.x]=sum(data[ctaid.x*256..+255])\n",
         N, threads, blocks);

  CUfunction fn;
  CK(cuModuleGetFunction(&fn, mod, "block_reduce"));

  rng_seed(0x1234abcd5678ef90ull);
  float* hd = malloc(N * 4);
  for (int i = 0; i < N; ++i) hd[i] = rnd_f32_in(-8.f, 8.f);
  float* ho = malloc(blocks * 4);
  memset(ho, 0x5a, blocks * 4);
  /* reference: thread 0 accumulates smem[0..ntid) sequentially in f32 */
  float* he = malloc(blocks * 4);
  for (int b = 0; b < blocks; ++b) {
    float acc = 0.f;
    for (int t = 0; t < threads; ++t) acc = acc + hd[b * threads + t];
    he[b] = acc;
  }

  CUdeviceptr dd, dout;
  CK(cuMemAlloc(&dd, N * 4));
  CK(cuMemAlloc(&dout, blocks * 4));
  CK(cuMemcpyHtoD(dd, hd, N * 4));
  CK(cuMemcpyHtoD(dout, ho, blocks * 4));

  int n = N;
  void* args[] = {&dd, &dout, &n};
  CK(cuLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, 0, args, 0));
  CUresult r = cuCtxSynchronize();
  if (r != CUDA_SUCCESS) {
    printf("  MISMATCH: launch/sync failed: %s (%s)\n", cuerr(r), cuerrstr(r));
    return 3;
  }
  CK(cuMemcpyDtoH(ho, dout, blocks * 4));

  int bad = 0, first = -1;
  for (int b = 0; b < blocks; ++b)
    if (memcmp(&ho[b], &he[b], 4) != 0) {
      if (first < 0) first = b;
      bad++;
    }
  if (!bad) {
    printf("  OK: %d blocks x %d threads, out[ctaid.x] = sum(data[ctaid.x*256..+255]) verified bit-exactly\n",
           blocks, threads);
    return 0;
  }
  printf("  MISMATCH: %d of %d block sums wrong, first at block %d\n", bad, blocks, first);
  dump_words("got", ho + first, 4, 0);
  dump_words("expected", he + first, 4, 0);
  return 1;
}

// ---------------------------------------------------------------------------
// kernel: fp16_reduce -- f16/bf16 pipeline over a block reduction
//   params: (const f16* data, f16* out_f16, bf16* out_bf16, u32 n)
//   BX=32 (sh[32] shared array), n must be a multiple of 32
// ---------------------------------------------------------------------------
#define FP16_BX 32
#define FP16_BLOCKS 8
#define FP16_N (FP16_BX * FP16_BLOCKS)

static uint16_t fp16_roundtrip(uint16_t h1) {
  /* cvt.f32.f16 ; cvt.rn.f16.f32 ; cvt.f32.f16 ; cvt.rn.bf16.f32 ;
     cvt.f32.bf16 ; cvt.rn.f16.f32 */
  float r6 = f16_bits_to_f32(h1);
  uint16_t h2 = f32_to_f16_bits(r6);
  float r7 = f16_bits_to_f32(h2);
  uint16_t h3 = f32_to_bf16_bits(r7);
  float r8 = bf16_bits_to_f32(h3);
  return f32_to_f16_bits(r8);
}

static void fp16_reference(const uint16_t* data, uint16_t* out16, uint16_t* outbf,
                           uint16_t* out16_dbl) {
  for (int b = 0; b < FP16_BLOCKS; ++b) {
    uint16_t sh[FP16_BX];
    for (int t = 0; t < FP16_BX; ++t) sh[t] = fp16_roundtrip(data[b * FP16_BX + t]);
    /* thread 0: h1=0 (f16), r7=0 (f32), i=0 ; loop i in [0, ntid.x) */
    uint16_t acc16 = 0;
    float acc32 = 0.f;
    for (int t = 0; t < FP16_BX; ++t) {
      acc16 = f16_add(acc16, sh[t]);
      acc32 = acc32 + f16_bits_to_f32(sh[t]);
    }
    out16[b] = f16_fma(acc16, acc16, acc16);          /* fma.rn.f16        */
    out16_dbl[b] = f16_fma_double_rounded(acc16, acc16, acc16);
    uint16_t bb = f32_to_bf16_bits(acc32);            /* cvt.rn.bf16.f32   */
    outbf[b] = bf16_add(bb, bb);                      /* add.rn.bf16       */
  }
}

static int test_fp16_reduce(CUmodule mod, int dataset) {
  printf("  config: blockDim=(%d,1,1) gridDim=(%d,1,1) n=%d   dataset=%s\n", FP16_BX,
         FP16_BLOCKS, FP16_N, dataset == 0 ? "small ints 1..4" : "random exact f16 in [-2,2)");

  CUfunction fn;
  CK(cuModuleGetFunction(&fn, mod, "fp16_reduce"));

  uint16_t* hd = malloc(FP16_N * 2);
  if (dataset == 0) {
    for (int i = 0; i < FP16_N; ++i) hd[i] = f32_to_f16_bits((float)(1 + (i % 4)));
  } else {
    rng_seed(0x0badc0de0badc0deull);
    for (int i = 0; i < FP16_N; ++i) hd[i] = rnd_f16_small();
  }
  uint16_t* h16 = malloc(FP16_BLOCKS * 2);
  uint16_t* hbf = malloc(FP16_BLOCKS * 2);
  memset(h16, 0x5a, FP16_BLOCKS * 2);
  memset(hbf, 0x5a, FP16_BLOCKS * 2);

  uint16_t* e16 = malloc(FP16_BLOCKS * 2);
  uint16_t* e16d = malloc(FP16_BLOCKS * 2);
  uint16_t* ebf = malloc(FP16_BLOCKS * 2);
  fp16_reference(hd, e16, ebf, e16d);

  int fma_differs = 0;
  for (int b = 0; b < FP16_BLOCKS; ++b)
    if (e16[b] != e16d[b]) fma_differs++;
  if (fma_differs)
    printf("  NOTE: fma.rn.f16 (source) and fma.rn.f32+f16-round (translated) disagree "
           "on %d/%d blocks for this data\n",
           fma_differs, FP16_BLOCKS);
  else
    printf("  NOTE: fma.rn.f16 (source semantics) and the translated "
           "fma.rn.f32+round agree on all %d blocks (data is small enough to be exact)\n",
           FP16_BLOCKS);

  CUdeviceptr dd, d16, dbf;
  CK(cuMemAlloc(&dd, FP16_N * 2));
  CK(cuMemAlloc(&d16, FP16_BLOCKS * 2));
  CK(cuMemAlloc(&dbf, FP16_BLOCKS * 2));
  CK(cuMemcpyHtoD(dd, hd, FP16_N * 2));
  CK(cuMemcpyHtoD(d16, h16, FP16_BLOCKS * 2));
  CK(cuMemcpyHtoD(dbf, hbf, FP16_BLOCKS * 2));

  unsigned int n = FP16_N;
  void* args[] = {&dd, &d16, &dbf, &n};
  CK(cuLaunchKernel(fn, FP16_BLOCKS, 1, 1, FP16_BX, 1, 1, 0, 0, args, 0));
  CUresult r = cuCtxSynchronize();
  if (r != CUDA_SUCCESS) {
    printf("  MISMATCH: launch/sync failed: %s (%s)\n", cuerr(r), cuerrstr(r));
    return 3;
  }
  CK(cuMemcpyDtoH(h16, d16, FP16_BLOCKS * 2));
  CK(cuMemcpyDtoH(hbf, dbf, FP16_BLOCKS * 2));

  int bad16 = 0, badbf = 0, first16 = -1, firstbf = -1;
  for (int b = 0; b < FP16_BLOCKS; ++b) {
    if (h16[b] != e16[b]) { if (first16 < 0) first16 = b; bad16++; }
    if (hbf[b] != ebf[b]) { if (firstbf < 0) firstbf = b; badbf++; }
  }
  if (!bad16 && !badbf) {
    printf("  OK: %d blocks, out_f16[block]=fma.f16(acc,acc,acc) and out_bf16[block]="
           "bf16(2*bf16(acc_f32)) verified bit-exactly\n",
           FP16_BLOCKS);
    return 0;
  }
  printf("  MISMATCH: out_f16 %d/%d wrong (first block %d), out_bf16 %d/%d wrong (first block %d)\n",
         bad16, FP16_BLOCKS, first16, badbf, FP16_BLOCKS, firstbf);
  if (bad16) {
    dump_words("out_f16 got", h16 + first16, 4, 1);
    dump_words("out_f16 expected", e16 + first16, 4, 1);
  }
  if (badbf) {
    dump_words("out_bf16 got", hbf + firstbf, 4, 1);
    dump_words("out_bf16 expected", ebf + firstbf, 4, 1);
  }
  return 1;
}

// ---------------------------------------------------------------------------
// kernel: gemm_tile -- 16x16 f16 tiled GEMM, one 16x16 tile per CTA
//   params: (const f16* A, const f16* B, float* C)
//   blockDim = (16,16); blockId = ctaid.y*nctaid.x + ctaid.x
//   A/B at element blockId*256 ; C at element blockId*131072
//        (source PTX: rd5 = blockId*512, then rd6 = rd5*1024)
// ---------------------------------------------------------------------------
static int test_gemm_tile(CUmodule mod, int gx, int gy, int dataset) {
  const int nblk = gx * gy;
  const size_t tile_elems = 256;             /* 16*16 f16 */
  /* Kernel contract (gemm_tile.ptx header): C lives at blockId*1024 bytes,
     i.e. 256 f32 per block. A/B tiles are contiguous at blockId*512 bytes. */
  const size_t c_off_elems = 256;
  const size_t c_elems = (size_t)nblk * c_off_elems;

  printf("  config: blockDim=(16,16) gridDim=(%d,%d) -> %d block(s); "
         "A/B tile at %zu f16 elems/block, C at %zu f32 elems/block   dataset=%s\n",
         gx, gy, nblk, tile_elems, c_off_elems,
         dataset == 0 ? "small ints" : "random exact f16 in [-2,2)");

  CUfunction fn;
  CK(cuModuleGetFunction(&fn, mod, "gemm_tile"));

  uint16_t* hA = malloc(nblk * tile_elems * 2);
  uint16_t* hB = malloc(nblk * tile_elems * 2);
  if (dataset == 0) {
    for (size_t i = 0; i < nblk * tile_elems; ++i) {
      hA[i] = f32_to_f16_bits((float)(1 + (i % 4)));
      hB[i] = f32_to_f16_bits((float)(1 + ((i / 16) % 2)));
    }
  } else {
    rng_seed(0xfeedfacecafebeefull);
    for (size_t i = 0; i < nblk * tile_elems; ++i) hA[i] = rnd_f16_small();
    for (size_t i = 0; i < nblk * tile_elems; ++i) hB[i] = rnd_f16_small();
  }

  float* hC = malloc(c_elems * 4);
  float* hE = malloc(c_elems * 4);
  for (size_t i = 0; i < c_elems; ++i) {
    uint32_t s = SENTINEL_F32;
    memcpy(&hC[i], &s, 4);
    hE[i] = 0.f;
  }

  /* reference: per block, cooperative f16 tile load + f16 dot product over K=16 */
  for (int b = 0; b < nblk; ++b) {
    const uint16_t* A = hA + (size_t)b * tile_elems;
    const uint16_t* B = hB + (size_t)b * tile_elems;
    uint16_t shA[256], shB[256];
    for (int row = 0; row < 16; ++row)
      for (int col = 0; col < 16; ++col) {
        shA[row * 16 + col] = A[row * 16 + col];
        shB[row * 16 + col] = B[row * 16 + col];
      }
    for (int row = 0; row < 16; ++row)
      for (int col = 0; col < 16; ++col) {
        uint16_t acc = 0;
        for (int k = 0; k < 16; ++k) {
          uint16_t prod = f16_mul(shA[row * 16 + k], shB[k * 16 + col]);
          acc = f16_add(acc, prod);
        }
        hE[(size_t)b * c_off_elems + row * 16 + col] = f16_bits_to_f32(acc);
      }
  }

  CUdeviceptr dA, dB, dC;
  CK(cuMemAlloc(&dA, nblk * tile_elems * 2));
  CK(cuMemAlloc(&dB, nblk * tile_elems * 2));
  CK(cuMemAlloc(&dC, c_elems * 4));
  CK(cuMemcpyHtoD(dA, hA, nblk * tile_elems * 2));
  CK(cuMemcpyHtoD(dB, hB, nblk * tile_elems * 2));
  CK(cuMemcpyHtoD(dC, hC, c_elems * 4));

  void* args[] = {&dA, &dB, &dC};
  CK(cuLaunchKernel(fn, gx, gy, 1, 16, 16, 1, 0, 0, args, 0));
  CUresult r = cuCtxSynchronize();
  if (r != CUDA_SUCCESS) {
    printf("  MISMATCH: launch/sync failed: %s (%s) -- the kernel faulted on the device\n",
           cuerr(r), cuerrstr(r));
    CUresult r2 = cuCtxSynchronize();
    printf("           follow-up sync: %s (%s)\n", cuerr(r2), cuerrstr(r2));
    return 3;
  }
  CK(cuMemcpyDtoH(hC, dC, c_elems * 4));

  /* only the first 256 entries of each block's C slice are written by the kernel */
  int bad = 0, first = -1, untouched = 0;
  for (int b = 0; b < nblk; ++b) {
    for (int i = 0; i < 256; ++i) {
      const float* g = &hC[(size_t)b * c_off_elems + i];
      const float* e = &hE[(size_t)b * c_off_elems + i];
      uint32_t sg;
      memcpy(&sg, g, 4);
      if (sg == SENTINEL_F32) { untouched++; continue; }
      if (memcmp(g, e, 4) != 0) {
        if (first < 0) first = b * 256 + i;
        bad++;
      }
    }
  }
  if (!bad && !untouched) {
    printf("  OK: %d block(s) x 256 outputs, C[row*16+col] = f16 dot product over K=16 "
           "verified bit-exactly\n",
           nblk);
    return 0;
  }
  printf("  MISMATCH: %d/%d outputs wrong, %d/%d outputs still hold the 0x%08x sentinel "
         "(never written by the kernel)\n",
         bad, nblk * 256, untouched, nblk * 256, SENTINEL_F32);
  if (first >= 0) {
    dump_words("C got", &hC[first], 4, 0);
    dump_words("C expected", &hE[first], 4, 0);
  }
  return 1;
}

// ---------------------------------------------------------------------------
// kernel: shfl_reduce -- out[ctaid.x] = warp-0 butterfly sum, written by tid 0
//
// Only thread 0 of each block writes, but its value is the full 32-lane
// butterfly reduction, so this test proves real cross-lane exchange: the
// scalar-identity fallback our host path uses would leave lane 0 holding just
// data[b*256]. The reference replays the tree order (offsets 16,8,4,2,1 with
// simultaneous exchange inside each step), which is the only order that is
// bit-exact for f32 rounding. N is a multiple of 32, so every warp is full
// and no lane is inactive during the exchange.
// ---------------------------------------------------------------------------
static int test_shfl_reduce(CUmodule mod) {
  const int threads = 256, blocks = 4, N = threads * blocks;
  printf("  config: N=%d blockDim=(%d,1,1) gridDim=(%d,1,1), out[ctaid.x]=butterfly sum of warp 0\n",
         N, threads, blocks);

  CUfunction fn;
  CK(cuModuleGetFunction(&fn, mod, "shfl_reduce"));

  float* hd = malloc(N * 4);
  for (int i = 0; i < N; ++i) hd[i] = (float)(((i % 9) + 1) * 0.5f);
  float* ho = malloc(blocks * 4);
  for (int i = 0; i < blocks; ++i) ho[i] = -1.f;
  float* he = malloc(blocks * 4);
  for (int b = 0; b < blocks; ++b) {
    float lane[32], next[32];
    for (int l = 0; l < 32; ++l) lane[l] = hd[b * threads + l];
    const int offs[5] = {16, 8, 4, 2, 1};
    for (int s = 0; s < 5; ++s) {
      for (int l = 0; l < 32; ++l) next[l] = lane[l] + lane[l ^ offs[s]];
      for (int l = 0; l < 32; ++l) lane[l] = next[l];
    }
    he[b] = lane[0];
  }

  CUdeviceptr dd, dout;
  CK(cuMemAlloc(&dd, N * 4));
  CK(cuMemAlloc(&dout, blocks * 4));
  CK(cuMemcpyHtoD(dd, hd, N * 4));
  CK(cuMemcpyHtoD(dout, ho, blocks * 4));

  int n = N;
  void* args[] = {&dd, &dout, &n};
  CK(cuLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, 0, args, 0));
  CUresult r = cuCtxSynchronize();
  if (r != CUDA_SUCCESS) {
    printf("  MISMATCH: launch/sync failed: %s (%s) -- the kernel faulted on the device\n",
           cuerr(r), cuerrstr(r));
    return 3;
  }
  CK(cuMemcpyDtoH(ho, dout, blocks * 4));

  int bad = 0;
  for (int b = 0; b < blocks; ++b)
    if (ho[b] != he[b]) {
      printf("  out[%d]: got %g want %g\n", b, ho[b], he[b]);
      bad = 1;
    }
  printf(bad ? "  MISMATCH\n" : "  OK: %d blocks, lane-0 butterfly sums verified bit-exactly\n",
         blocks);
  return bad;
}

// ---------------------------------------------------------------------------
// kernel: atom_cas -- per-thread slots for cas/exch/min/max with old-value
// buffers. Every thread owns its slot (g = ctaid*ntid+tid), so no two threads
// race and the outcome is fully deterministic: both the returned old values
// and the final memory contents are checked.
// ---------------------------------------------------------------------------
static int test_atom_cas(CUmodule mod) {
  const int threads = 256, blocks = 1, N = threads * blocks;
  printf("  config: N=%d blockDim=(%d,1,1) gridDim=(%d,1,1), private slots\n",
         N, threads, blocks);

  CUfunction fn;
  CK(cuModuleGetFunction(&fn, mod, "atom_cas"));

  uint32_t *hm[4], *ho[4];
  const uint32_t init[4] = {42, 11, 50, 20};
  const uint32_t want_mem[4] = {100, 55, 10, 90};
  for (int k = 0; k < 4; ++k) {
    hm[k] = malloc(N * 4);
    ho[k] = malloc(N * 4);
    for (int i = 0; i < N; ++i) {
      hm[k][i] = init[k];
      ho[k][i] = 0xdeadbeef;
    }
  }

  CUdeviceptr dm[4], dout[4];
  for (int k = 0; k < 4; ++k) {
    CK(cuMemAlloc(&dm[k], N * 4));
    CK(cuMemAlloc(&dout[k], N * 4));
    CK(cuMemcpyHtoD(dm[k], hm[k], N * 4));
    CK(cuMemcpyHtoD(dout[k], ho[k], N * 4));
  }

  int n = N;
  void* args[] = {&dm[0], &dm[1], &dm[2], &dm[3],
                  &dout[0], &dout[1], &dout[2], &dout[3], &n};
  CK(cuLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, 0, args, 0));
  CUresult r = cuCtxSynchronize();
  if (r != CUDA_SUCCESS) {
    printf("  MISMATCH: launch/sync failed: %s (%s) -- the kernel faulted on the device\n",
           cuerr(r), cuerrstr(r));
    return 3;
  }
  for (int k = 0; k < 4; ++k) {
    CK(cuMemcpyDtoH(hm[k], dm[k], N * 4));
    CK(cuMemcpyDtoH(ho[k], dout[k], N * 4));
  }

  static const char* const what[4] = {"cas 42->100", "exch 11->55",
                                      "min.s32 50/10", "max.u32 20/90"};
  int bad = 0;
  for (int k = 0; k < 4; ++k)
    for (int i = 0; i < N; ++i) {
      if (ho[k][i] != init[k]) {
        if (bad < 4) printf("  %s: old[%d] got %u want %u\n", what[k], i, ho[k][i], init[k]);
        bad = 1;
      }
      if (hm[k][i] != want_mem[k]) {
        if (bad < 4) printf("  %s: mem[%d] got %u want %u\n", what[k], i, hm[k][i], want_mem[k]);
        bad = 1;
      }
    }
  printf(bad ? "  MISMATCH\n" : "  OK: %d threads, old values and final memory verified exactly\n",
         N);
  return bad;
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  if (argc < 3) {
    printf("usage: %s <file.ptx> <vadd|block_reduce|fp16_reduce|gemm_tile|shfl_reduce|atom_cas|all>\n", argv[0]);
    return 2;
  }
  const char* ptx = argv[1];
  const char* kname = argv[2];

  FILE* f = fopen(ptx, "rb");
  if (!f) {
    printf("FAIL open %s\n", ptx);
    return 2;
  }
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  char* buf = malloc(sz + 1);
  if (fread(buf, 1, sz, f) != (size_t)sz) {
    printf("FAIL read %s\n", ptx);
    return 2;
  }
  buf[sz] = 0;
  fclose(f);

  // Environment guards: this test needs a real NVIDIA GPU on the host.
  // Without one (CI machines, AMD/Intel-only boxes) it reports SKIP instead
  // of failing. CTest matches these markers via SKIP_REGULAR_EXPRESSION.
  if (cuInit(0) != CUDA_SUCCESS) {
    printf("NO-DEVICE: cuInit failed, no NVIDIA driver on this host\n");
    return 0;
  }
  CUdevice dev;
  if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) {
    printf("NO-DEVICE: no CUDA-capable device on this host\n");
    return 0;
  }
  {
    int ccmaj = 0, ccmin = 0;
    cuDeviceGetAttribute(&ccmaj, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&ccmin, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    // The PTX below is built for sm_75; the driver JIT cannot retarget it to
    // an older architecture, so such a card is an environment limitation.
    if (ccmaj < 7 || (ccmaj == 7 && ccmin < 5)) {
      printf("SKIP-OLD-ARCH: device cc %d.%d < 7.5, PTX targets sm_75\n", ccmaj, ccmin);
      return 0;
    }
  }
  char name[128];
  CK(cuDeviceGetName(name, 128, dev));
  int ccmaj = 0, ccmin = 0, drv = 0, nsm = 0;
  CK(cuDeviceGetAttribute(&ccmaj, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
  CK(cuDeviceGetAttribute(&ccmin, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
  CK(cuDeviceGetAttribute(&nsm, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
  CK(cuDriverGetVersion(&drv));
  printf("device: %s, compute capability %d.%d, %d SMs, driver API %d\n", name, ccmaj, ccmin,
         nsm, drv);
  printf("ptx   : %s (%ld bytes)\n", ptx, sz);

  CUcontext ctx;
  if (make_ctx(&ctx, dev)) return 1;
  CUmodule mod;
  CUresult mr = cuModuleLoadData(&mod, buf);
  if (mr != CUDA_SUCCESS) {
    printf("FAIL cuModuleLoadData: %s (%s)\n", cuerr(mr), cuerrstr(mr));
    return 2;
  }
  printf("module loaded and JIT-compiled for the device\n\n");

  model_crosscheck();
  printf("f16/bf16 reference model: %u checks, %u disagreements with the compiler's _Float16\n\n",
         g_model_checked, g_model_mismatch);
  if (g_model_mismatch) printf("  WARNING: the bit-level f16 model and _Float16 DISAGREE\n");

  int rc = 0;
  int all = strcmp(kname, "all") == 0;

  if (all || strcmp(kname, "vadd") == 0) {
    printf("== vadd ==\n");
    rc = test_vadd(mod);
    printf("   => %s\n\n", rc == 0 ? "OK" : (rc == 3 ? "DEVICE FAULT" : "MISMATCH"));
    if (rc && g_exit_code == 0) g_exit_code = rc;
  }
  if (all || strcmp(kname, "block_reduce") == 0) {
    printf("== block_reduce ==\n");
    rc = test_block_reduce(mod);
    printf("   => %s\n\n", rc == 0 ? "OK" : (rc == 3 ? "DEVICE FAULT" : "MISMATCH"));
    if (rc && g_exit_code == 0) g_exit_code = rc;
  }
  if (all || strcmp(kname, "fp16_reduce") == 0) {
    printf("== fp16_reduce (dataset A) ==\n");
    rc = test_fp16_reduce(mod, 0);
    printf("   => %s\n\n", rc == 0 ? "OK" : (rc == 3 ? "DEVICE FAULT" : "MISMATCH"));
    if (rc && g_exit_code == 0) g_exit_code = rc;
    if (!all) goto done;
    printf("== fp16_reduce (dataset B) ==\n");
    rc = test_fp16_reduce(mod, 1);
    printf("   => %s\n\n", rc == 0 ? "OK" : (rc == 3 ? "DEVICE FAULT" : "MISMATCH"));
    if (rc && g_exit_code == 0) g_exit_code = rc;
  }
  if (all || strcmp(kname, "gemm_tile") == 0) {
    printf("== gemm_tile (grid 1x1, dataset A) ==\n");
    rc = test_gemm_tile(mod, 1, 1, 0);
    printf("   => %s\n\n", rc == 0 ? "OK" : (rc == 3 ? "DEVICE FAULT" : "MISMATCH"));
    if (rc && g_exit_code == 0) g_exit_code = rc;
    if (!all) goto done;
    printf("== gemm_tile (grid 1x1, dataset B) ==\n");
    rc = test_gemm_tile(mod, 1, 1, 1);
    printf("   => %s\n\n", rc == 0 ? "OK" : (rc == 3 ? "DEVICE FAULT" : "MISMATCH"));
    if (rc && g_exit_code == 0) g_exit_code = rc;
    printf("== gemm_tile (grid 2x1, dataset A) ==\n");
    rc = test_gemm_tile(mod, 2, 1, 0);
    printf("   => %s\n\n", rc == 0 ? "OK" : (rc == 3 ? "DEVICE FAULT" : "MISMATCH"));
    if (rc && g_exit_code == 0) g_exit_code = rc;
  }
  if (all || strcmp(kname, "shfl_reduce") == 0) {
    printf("== shfl_reduce ==\n");
    rc = test_shfl_reduce(mod);
    printf("   => %s\n\n", rc == 0 ? "OK" : (rc == 3 ? "DEVICE FAULT" : "MISMATCH"));
    if (rc && g_exit_code == 0) g_exit_code = rc;
  }
  if (all || strcmp(kname, "atom_cas") == 0) {
    printf("== atom_cas ==\n");
    rc = test_atom_cas(mod);
    printf("   => %s\n\n", rc == 0 ? "OK" : (rc == 3 ? "DEVICE FAULT" : "MISMATCH"));
    if (rc && g_exit_code == 0) g_exit_code = rc;
  }
  if (!all && strcmp(kname, "fp16_reduceB") == 0) {
    printf("== fp16_reduce (dataset B only) ==\n");
    rc = test_fp16_reduce(mod, 1);
    printf("   => %s\n\n", rc == 0 ? "OK" : (rc == 3 ? "DEVICE FAULT" : "MISMATCH"));
    if (rc && g_exit_code == 0) g_exit_code = rc;
    goto done;
  }
  if (!all && strcmp(kname, "gemm_tileB") == 0) {
    printf("== gemm_tile (grid 1x1, dataset B only) ==\n");
    rc = test_gemm_tile(mod, 1, 1, 1);
    printf("   => %s\n\n", rc == 0 ? "OK" : (rc == 3 ? "DEVICE FAULT" : "MISMATCH"));
    if (rc && g_exit_code == 0) g_exit_code = rc;
    goto done;
  }
  if (!all && strcmp(kname, "gemm_tile2") == 0) {
    printf("== gemm_tile (grid 2x1, dataset A only) ==\n");
    rc = test_gemm_tile(mod, 2, 1, 0);
    printf("   => %s\n\n", rc == 0 ? "OK" : (rc == 3 ? "DEVICE FAULT" : "MISMATCH"));
    if (rc && g_exit_code == 0) g_exit_code = rc;
    goto done;
  }
  if (!all && strcmp(kname, "vadd") && strcmp(kname, "block_reduce") &&
      strcmp(kname, "fp16_reduce") && strcmp(kname, "gemm_tile") &&
      strcmp(kname, "shfl_reduce") && strcmp(kname, "atom_cas") &&
      strcmp(kname, "fp16_reduceB") && strcmp(kname, "gemm_tileB") &&
      strcmp(kname, "gemm_tile2")) {
    printf("unknown kernel '%s'\n", kname);
    return 2;
  }

done:
  return g_exit_code;
}
