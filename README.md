<h1 align="center">ROUGE-V</h1>

<p align="center">A compiler that translates CUDA PTX into native code for RISC-V.</p>

<p align="center">
<a href="https://github.com/MrModelOS/ROUGE-V/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/MrModelOS/ROUGE-V/ci.yml?style=flat-square&branch=main"></a>
<img alt="Tests" src="https://img.shields.io/badge/tests-24%2F24%20passed-brightgreen?style=flat-square">
<img alt="Language" src="https://img.shields.io/badge/language-C%2B%2B-blue?style=flat-square">
<img alt="License" src="https://img.shields.io/badge/license-Apache--2.0%20WITH%20LLVM--exception-blue?style=flat-square">
</p>

<p align="center">
<a href="README.md">English</a> ·
<a href="README.ru.md">Русский</a> ·
<a href="README.zh.md">中文</a>
</p>

---

`nvcc` does not produce machine code for a GPU. It produces **PTX** — a textual
assembly with a published specification. Anything that can read PTX can compile
it for a different target.

ROUGE-V does that: PTX → LLVM IR → native code, for RISC-V and its vector
extension `rv64gcv`. It is a compiler, not an emulator — the translated kernel
is compiled, not interpreted instruction by instruction.

### Install

```sh
git clone https://github.com/MrModelOS/ROUGE-V.git
cd ROUGE-V
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Requires CMake ≥ 3.20, a C++17 compiler, and `clang`. `nvcc` is optional — it
enables the vendor-PTX test.

### Use

No build needed — two binaries with no dependencies beyond system libc:

```sh
curl -fsSL https://github.com/MrModelOS/ROUGE-V/releases/download/v0.1/rouge-v-0.1-linux-x86_64.tar.gz | tar xz -C ~/.local
export PATH="$HOME/.local/bin:$PATH"
ptx2ir kernel.ptx kernel.ll
```

Run it on an NVIDIA GPU (sm_75+, driver only, no toolkit needed at run time):

```sh
rouge-run --target nvptx64-nvidia-cuda kernel.ptx --emit-ptx kernel_gpu.ptx
# then fill the 3 EDIT points in software/rouge-compiler/examples/run_on_gpu.c
# (PTX path, kernel name, buffers/launch) and build it against libcuda
```

(Arch Linux: `packaging/PKGBUILD` for a manual build. `packaging/install.sh` does the above.)

The question a user has is "will you translate my kernel?". One command answers it:

```sh
./build/rouge-compiler/rouge-run kernel.ptx
```

```
input: kernel.ptx
kernels found: 1
  @gemm_tile  params=3  instructions=67
  shared memory: 1024 bytes in 2 variable(s)
    shA          @   0     512 bytes  (.b16 x 256)
    shB          @ 512     512 bytes  (.b16 x 256)
translation: ok (10869 bytes of LLVM IR)

OK
```

Outside the supported subset you get a refusal, not a wrong answer:

```
rouge-run: REFUSED: warp-cooperative 'wmma.mma.sync.aligned.m16n16k8.row.col.f32.f16.f16.f32' is not lowered (wmma is a different matrix API; movmatrix is a warp-wide permutation): refused loudly rather than lowered as something else
This instruction is outside the supported subset. Nothing was emitted;
nothing was approximated.
```

Emit a native object, or inspect the IR directly:

```sh
./build/rouge-compiler/rouge-run kernel.ptx -o kernel.o --print-ir
./build/rouge-compiler/ptx2ir kernel.ptx kernel.ll
```

Target RISC-V instead of the host:

```sh
clang --target=riscv64-unknown-elf -march=rv64gcv -c kernel.ll -o kernel-rv64.o
```

The triple itself is chosen on the translator side with `--target`, which is
just a prefix of `ptx2ir`'s arguments — the rest of the command is unchanged.
The same PTX therefore builds for four platforms:

```sh
./build/rouge-compiler/ptx2ir --target amdgcn-amd-amdhsa kernel.ptx kernel-amd.ll
clang --target=amdgcn-amd-amdhsa -mcpu=gfx1100 -c kernel-amd.ll -o kernel-amd.o
```

| `--target` | Platform | Backend flags | What is actually verified |
|---|---|---|---|
| *(default)* `x86_64-pc-linux-gnu` | host x86-64 | — | compiles **and executes** — the `aot_native_*` ctest suite |
| `riscv64-unknown-elf` | RISC-V + Vector | `-march=rv64gcv` | compiles to a native object — `ctest compiler_rvv_backend_*`; a barrier-free kernel also **runs** under `qemu-riscv64` — `ctest compiler_rv_exec` (one lane at a time; kernels with `bar.sync` need real threads and are not claimed) |
| `amdgcn-amd-amdhsa` | AMD RDNA 3 | `-mcpu=gfx1100` | compiles to a native object — never executed |
| `nvptx64-nvidia-cuda` | NVIDIA | `-march=sm_75` | compiles **and executes on a real GPU** — see below |

All fifteen canonical kernels in `software/rouge-compiler/tests/kernels/` — `vadd`,
`block_reduce`, `atomic_reduce`, `fp16_reduce`, `fp8_cvt`, `gemm_tile`, `shfl_reduce`,
`atom_cas`, `shfl_pred`, `vote_pred`, `shfl_pred_all`, `mma_tile`, `mma_forms`,
`mma_pipe`, `ldmatrix_tile` — build for all four triples.

### Executed on real hardware

The NVIDIA kernels do not just compile: `ctest compiler_nvptx_gpu_exec*`
loads the generated PTX through the CUDA Driver API (driver-side JIT) on a
real GeForce MX450 (sm_75) and checks every output bit-exactly against a host
reference — all ten canonical kernels, two datasets for `fp16_reduce`/`gemm_tile`,
plus a two-block grid proving the multi-block path. `compiler_nvcc_gpu_exec`
additionally runs genuine vendor output (`nvcc real_nvcc.cu`) end to end. On
machines without an sm_75+ NVIDIA GPU the tests report SKIP, never failure.

`ldmatrix` is the first warp-collective matrix op that ran on real silicon
here (sm_75 — it is shared-memory traffic, no tensor cores needed). `mma` and
`stmatrix` assemble with `ptxas` at sm_80/sm_90, but this card has no tensor
cores, so their *execution* remains compile-verified only.

The AMD row stays a **compile-time** claim: no AMD device is involved anywhere
in the project. Neither row says anything about performance — only bit-identity
against the host reference was measured. The target changes the address spaces
in the emitted IR, which is what lets those backends emit `global_load`/
`global_store` instead of scalar accesses — see
[docs/06-compiler-architecture.md](docs/06-compiler-architecture.md).

Where the PTX comes from:

```sh
nvcc -arch=sm_75 -ptx kernel.cu -o kernel.ptx   # build from CUDA source
cuobjdump -ptx ./my_app > kernel.ptx             # extract from a fatbin
```

[QUICKSTART.md](QUICKSTART.md) covers the rest, including the minimal host
driver needed to launch a translated kernel.

```
$ ctest --test-dir build
...
 7/56 Test  #7: aot_native_block_reduce ..................   Passed    0.08 sec
 8/56 Test  #8: aot_native_atomic_reduce .................   Passed    0.07 sec
23/56 Test #23: compiler_foreign_ptx .....................   Passed    0.36 sec
33/56 Test #33: compiler_nvptx_gpu_exec ..................   Passed    7.73 sec
35/56 Test #35: compiler_nvcc_gpu_exec ...................   Passed    0.52 sec
100% tests passed out of 56
```

> [!NOTE]
> `compiler_nvcc_ptx` compiles real CUDA C with the vendor's own `nvcc` and
> pushes the resulting PTX through the whole pipeline. It skips when `nvcc` is
> not installed.

> [!IMPORTANT]
> Instructions outside the supported subset fail with an error naming the
> instruction. There is no silent miscompilation — a wrong answer is worse than
> a refusal.

### Components

| Component | What it is |
|---|---|
| [`rouge-ptx`](software/rouge-ptx/) | Shared PTX front-end: parser + reference interpreter |
| [`rouge-compiler`](software/rouge-compiler/) | `ptx2ir`, the AOT translator; host runtime; MLIR contour |
| [`rouge-cuda`](software/rouge-cuda/) | CUDA Driver/Runtime API interposition and fallback execution path |
| [`rouge-numerics`](software/rouge-numerics/) | Own open math: GEMM (f32/f64/f16/bf16/FP8/complex), convolution, attention, FFT, sparse, RNG, INT8 quantization — plus cuBLAS/cuDNN compat shims over that code |
| [`rouge-render`](software/rouge-render/) | Own renderer: path tracer, à-trous denoiser, temporal upscaler — the open path where DLSS/OptiX would be |

Both execution paths are built on the same parser. That is deliberate: one
grammar, one set of rules for registers, parameters, `.shared` layout and
labels, so the paths cannot drift apart. The suite is built around the
requirement that they agree bit-for-bit on every kernel.

### Own math instead of NVIDIA's libraries

`rouge-numerics` is the "replace closed with open" half of the project: our
own kernels (not wrappers) and, on top of them, cuBLAS/cuDNN-API shims so
code written against those APIs runs here. Measured against the real open
libraries on the same machine (i5-1135G7, every number from a run):

| Comparison | Ours | Open reference | Verdict |
|---|---|---|---|
| GEMM f32 128³, 1 thread | ~88 GFLOP/s | OpenBLAS 0.3.34: ~104 | we lose by 1.2× |
| Convolution 3×3, 7 shapes | 0.21–4.55 ms | oneDNN 3.11.3: 3.1–194.8 ms | we are 12–43× faster |
| FFT 4096 | 0.077 ms | FFTW 3.7.11: 0.006 ms | we lose by 13× |

The losses are stated in the README of that component, not hidden. The
`foreign_cublas_program` test closes the loop: a program linked against
**NVIDIA's own libcublas** computes on our shim, on a machine with no
NVIDIA present, and prints exactly what the real cuBLAS printed on the GPU.

### Inference and rendering, the open way

Two pieces of the stack that exist so that nothing has to be closed:

- **Inference** (`rouge-numerics`): INT8 quantization in three schemes
  (per-tensor, per-channel, asymmetric per-group with an integer zero point),
  an **exact** integer GEMM (int32 accumulation), GQA/MQA attention with a
  sliding window and an FP8 KV cache in our own e4m3 format, and a full
  pre-norm transformer block (RMSNorm → RoPE → attention → SwiGLU MLP) whose
  projections run on our blocked GEMM. On the test machine the block runs at
  36.6 GFLOP/s; switching the projections from a naive triple loop to our GEMM
  made it **14× faster** (566 → 40.5 ms).
- **Rendering** (`rouge-render`): a path tracer (next-event estimation + MIS,
  so the reference image is unbiased), an à-trous wavelet denoiser and a
  temporal upscaler with jitter-aware reconstruction, history reprojection
  and variance clipping. Measured against a converged reference of the same
  render:

  | Step | Ours | Baseline | Verdict |
  |---|---|---|---|
  | Denoise a 4-sample frame | PSNR 28.79, SSIM 0.927 | noisy: 27.91 / 0.873 | +0.88 dB, +0.054 SSIM |
  | Upscale 2× (static camera) | PSNR 25.08, SSIM 0.816 | bilinear: 21.60 / 0.693 | +3.5 dB, +0.12 SSIM |
  | Upscale 2× (moving camera) | PSNR 25.25, SSIM 0.826 | bilinear: 21.60 / 0.693 | +3.7 dB, +0.13 SSIM |

  This is **not** DLSS: DLSS reconstructs with a closed neural network, ours
  is a classical pipeline (bicubic + reprojection + clipping). The point is
  that the whole path is open, ours, and measurable.

### Supported PTX

| Group | Instructions |
|---|---|
| Parameters | `ld.param.{u32,u64}` and signed/bit variants |
| Moves | `mov.*`, `cvta.to.global.*`, `cvta.to.shared.*` |
| Arithmetic | `add` `sub` `mul` `div` `rem`, `mad.lo` `mad.hi`, `mul.wide.*` |
| Bitwise / shift | `and` `or` `xor` `not`, `shl`, `shr.{u,s}`, `bfi` `bfe`, `prmt.b32` |
| Byte dot products | `dp4a.{s,u}32.{s,u}32`, `dp2a.{lo,hi}.{s,u}32.{s,u}32` |
| Compare | `setp.{ge,gt,lt,le,eq,ne}.{u32,s32,f32}` |
| Select | `selp.*`, `slct.*`, predicate `and` `or` `not` |
| Min / max / unary | `min` `max.*`, `neg`, `abs`, `popc`, `clz`, `bfind.shiftamt` |
| Floating point | `f16` / `bf16` arithmetic, `fma.rn`, `sqrt`, `rcp`, `rsqrt`, `fneg` |
| Memory | `ld`/`st` global and shared: `f32` `u32` `u64` `u8`/`s8` `u16`/`s16` `f16` `bf16` `v2.f32` |
| Shared memory | `.shared` declarations, `cvta.to.shared`, `[reg+offset]` indexing |
| Synchronization | `bar.sync` → runtime barrier |
| Warp | `shfl.{idx,bfly,up,down}` (incl. predicated destinations), `vote.*`, `activemask` |
| Atomics | `atom.add` `cas` `exch` `min` `max`, `red.add.*` → `atomicrmw` / `cmpxchg` |
| Matrix | `mma.sync.aligned` (.row.col m16n8k16 f16/bf16/e4m3/e5m2/u8/s8 → f32/f16/s32, m16n8k8 f16/bf16 → f32/f16), `ldmatrix`/`stmatrix.sync.aligned.m8n8.x{1,2,4}[.trans].b16` |

`wmma.*`, `movmatrix`, `mma.sp` (sparse) and `cvt.sat.*` rounding modes are
not implemented and are reported as errors — refused loudly, never approximated.

### Documentation

[docs/06-compiler-architecture.md](docs/06-compiler-architecture.md) — launch
descriptor, shared-memory layout, `bar.sync` model, atomics, FP16/BF16 lowering,
and the planned SIMT → RVV path.

### Building on ROUGE-V

If you maintain a project that uses ROUGE-V — a fork, a binding, a backend —
please note in your README that it is not built by the ROUGE-V team and is not
affiliated with it.

### Legal

ROUGE-V contains no NVIDIA source code, headers, or binaries, and performs no
reverse engineering of them. The PTX it compiles is produced by the user's own
`nvcc` installation, under the user's NVIDIA license, from a published
specification. The CUDA API headers under `software/rouge-cuda/include/` are
written from scratch against public API documentation. The clean-room rule is
enforced in [CONTRIBUTING.md](CONTRIBUTING.md).

ROUGE-V is not affiliated with, endorsed by, or supported by NVIDIA. CUDA and
other trademarks are the property of their respective owners and are used here
only to describe compatibility.

### Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) first — it covers the build, the test
workflow, and the clean-room requirement that applies to all contributions.
Issues and pull requests are open.

### Acknowledgements

- **LLVM / MLIR** — the IR, backend, and pass infrastructure this is written against
- **RISC-V International** — the `rv64gcv` ISA and Vector Extension specifications
- **NVIDIA** — the published PTX ISA and CUDA API documentation, which define the input format

### License

Apache 2.0 WITH LLVM-exception — the same license as LLVM, so this code can live
inside LLVM and MLIR without friction.
