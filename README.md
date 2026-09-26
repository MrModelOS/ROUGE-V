<h1 align="center">ROUGE-V</h1>

<p align="center">A compiler that translates CUDA PTX into native code for RISC-V.</p>

<p align="center">
<a href="https://github.com/MrModelOS/ROUGE-V/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/MrModelOS/ROUGE-V/ci.yml?style=flat-square&branch=main"></a>
<img alt="Tests" src="https://img.shields.io/badge/tests-24%2F24%20passed-brightgreen?style=flat-square">
<img alt="Language" src="https://img.shields.io/badge/language-C%2B%2B-blue?style=flat-square">
<img alt="License" src="https://img.shields.io/badge/license-Apache--2.0%20WITH%20LLVM--exception-blue?style=flat-square">
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

```sh
./build/rouge-compiler/ptx2ir \
    software/rouge-compiler/tests/kernels/vadd.ptx /tmp/vadd.ll

clang -O2 -c /tmp/vadd.ll -o /tmp/vadd.o                  # host
clang --target=riscv64-unknown-elf -march=rv64gcv \
      -c /tmp/vadd.ll -o /tmp/vadd-rv64.o                 # RISC-V + Vector
```

```
$ ctest --test-dir build
...
 5/16 aot_native_block_reduce ..............   Passed   smem 1024B 16x256
 6/16 aot_native_atomic_reduce .............   Passed   sum=4608 count=4096
 7/16 aot_native_fp16_reduce ...............   Passed   32x32 f16/bf16
 8/16 aot_native_gemm_tile .................   Passed   2x2 blocks 16x16
13/16 compiler_nvcc_ptx ....................   Passed   nvcc -> ptx2ir -> clang
100% tests passed out of 16
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

Both execution paths are built on the same parser. That is deliberate: one
grammar, one set of rules for registers, parameters, `.shared` layout and
labels, so the paths cannot drift apart. The suite is built around the
requirement that they agree bit-for-bit on every kernel.

### Supported PTX

| Group | Instructions |
|---|---|
| Parameters | `ld.param.{u32,u64}` and signed/bit variants |
| Moves | `mov.*`, `cvta.to.global.*`, `cvta.to.shared.*` |
| Arithmetic | `add` `sub` `mul` `div` `rem`, `mad.lo` `mad.hi`, `mul.wide.*` |
| Bitwise / shift | `and` `or` `xor` `not`, `shl`, `shr.{u,s}`, `bfi` `bfe` |
| Compare | `setp.{ge,gt,lt,le,eq,ne}.{u32,s32,f32}` |
| Select | `selp.*`, `slct.*`, predicate `and` `or` `not` |
| Min / max / unary | `min` `max.*`, `neg`, `abs`, `popc`, `clz`, `bfind.shiftamt` |
| Floating point | `f16` / `bf16` arithmetic, `fma.rn`, `sqrt`, `rcp`, `rsqrt`, `fneg` |
| Memory | `ld`/`st` global and shared: `f32` `u32` `u64` `u8`/`s8` `u16`/`s16` `f16` `bf16` `v2.f32` |
| Shared memory | `.shared` declarations, `cvta.to.shared`, `[reg+offset]` indexing |
| Synchronization | `bar.sync` → runtime barrier |
| Warp | `shfl.{idx,bfly,up,down}` (incl. predicated destinations), `vote.*`, `activemask` |
| Atomics | `atom.add` `cas` `exch` `min` `max`, `red.add.*` → `atomicrmw` / `cmpxchg` |

Tensor instructions (`mma`), warp `f16x2` vector forms, and `cvt.sat` rounding
modes are not implemented yet and are reported as errors.

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
