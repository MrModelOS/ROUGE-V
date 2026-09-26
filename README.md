# ROUGE-V

A compiler that translates CUDA PTX into native code for RISC-V.

[![CI](https://github.com/MrModelOS/ROUGE-V/actions/workflows/ci.yml/badge.svg)](https://github.com/MrModelOS/ROUGE-V/actions/workflows/ci.yml)
![License](https://img.shields.io/badge/license-Apache--2.0%20WITH%20LLVM--exception-blue)

`nvcc` does not produce machine code for a GPU. It produces **PTX** — a textual
assembly with a published specification. Anything that can read PTX can compile
it for a different target.

ROUGE-V does exactly that: PTX → LLVM IR → native code, for RISC-V and the
vector extension `rv64gcv`. It is a compiler, not an emulator — the translated
kernel is compiled, not interpreted instruction by instruction.

## Try it

```sh
git clone https://github.com/MrModelOS/ROUGE-V.git
cd ROUGE-V
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Translate a kernel by hand:

```sh
./build/rouge-compiler/ptx2ir \
    software/rouge-compiler/tests/kernels/vadd.ptx /tmp/vadd.ll

clang -O2 -c /tmp/vadd.ll -o /tmp/vadd.o                       # host
clang --target=riscv64-unknown-elf -march=rv64gcv \
      -c /tmp/vadd.ll -o /tmp/vadd-rv64.o                      # RISC-V + Vector
```

## Components

| Component | What it is |
|---|---|
| [`rouge-ptx`](software/rouge-ptx/) | Shared PTX front-end: parser + reference interpreter |
| [`rouge-compiler`](software/rouge-compiler/) | `ptx2ir`, the AOT translator; host runtime; MLIR contour |
| [`rouge-cuda`](software/rouge-cuda/) | CUDA Driver/Runtime API interposition and fallback execution path |

The AOT path and the interpreter must agree bit-for-bit on every kernel. That
constraint is what the test suite is built around.

## Supported PTX subset

| Group | Instructions |
|---|---|
| Parameters | `ld.param.{u32,u64}` (+ signed/bit variants) |
| Moves | `mov.*`, `cvta.to.global.*`, `cvta.to.shared.*` |
| Arithmetic | `add`/`sub`/`mul`/`div`/`rem`, `mad.lo`/`mad.hi`, `mul.wide.*` |
| Bitwise / shift | `and`/`or`/`xor`/`not`, `shl`, `shr.{u,s}`, `bfi`/`bfe` |
| Compare | `setp.{ge,gt,lt,le,eq,ne}.{u32,s32,f32}` |
| Select | `selp.*`, `slct.*`, predicate `and`/`or`/`not` |
| Min / max / unary | `min`/`max.*`, `neg`, `abs`, `popc`, `clz`, `bfind.shiftamt` |
| Floating point | `f16`/`bf16` arithmetic, `fma.rn`, `sqrt`, `rcp`, `rsqrt`, `fneg` |
| Memory | `ld`/`st` global and shared: `f32`, `u32`, `u64`, `u8`/`s8`, `u16`/`s16`, `f16`/`bf16`, `v2.f32` |
| Shared memory | `.shared` declarations, `cvta.to.shared`, `[reg+offset]` indexing |
| Synchronization | `bar.sync` → runtime barrier |
| Warp | `shfl.{idx,bfly,up,down}` (incl. predicated destinations), `vote.*`, `activemask` |
| Atomics | `atom.add`/`cas`/`exch`/`min`/`max`, `red.add.*` → LLVM `atomicrmw` / `cmpxchg` |

Unsupported instructions produce an error naming the instruction. There is no
silent miscompilation — a wrong answer is worse than a refusal.

## Tests

24 tests, all passing:

| Suite | Count | What it covers |
|---|---|---|
| `rouge-compiler` | 16 | structural IR checks, AOT execution, vendor PTX, RISC-V backend |
| `rouge-cuda` | 8 | same kernels through the Driver API and the interpreter |

`compiler_nvcc_ptx` is the one that matters most: it compiles real CUDA C with
the vendor's own `nvcc`, takes the PTX it produces, and pushes it through
`ptx2ir` → `clang`. It skips if `nvcc` is not installed.

```
$ ctest --test-dir build
...
13/16 compiler_nvcc_ptx ....................   Passed   nvcc -> ptx2ir -> clang
100% tests passed out of 16
```

## Architecture

[docs/06-compiler-architecture.md](docs/06-compiler-architecture.md) — the
launch descriptor, shared-memory layout, the `bar.sync` model, atomics, FP16 /
BF16 lowering, and the planned SIMT → RVV path.

## Legal

ROUGE-V contains no NVIDIA source code, headers, or binaries, and performs no
reverse engineering of them. The PTX it compiles is produced by the user's own
`nvcc` installation, under the user's NVIDIA license, from a published
specification. The CUDA API headers under `software/rouge-cuda/include/` are
written from scratch against public API documentation. The clean-room rule is
enforced in [CONTRIBUTING.md](CONTRIBUTING.md).

ROUGE-V is not affiliated with, endorsed by, or supported by NVIDIA. CUDA and
other trademarks are the property of their respective owners and are used here
only to describe compatibility.

## Contributing

Issues and pull requests are open. Good first targets are listed in the
roadmap of the compiler component; `atom.cas` coverage and `f16x2` warp
variants are the obvious ones. Read [CONTRIBUTING.md](CONTRIBUTING.md) first —
it describes the build, the test workflow, and the clean-room requirement that
applies to all contributions.

## Acknowledgements

ROUGE-V builds on work of others, principally:

- **LLVM / MLIR** — IR, backend, and the pass infrastructure this is written against
- **RISC-V International** — the `rv64gcv` ISA and Vector Extension specifications
- **NVIDIA** — the published PTX ISA and CUDA API documentation, which define the input format

## License

Apache 2.0 WITH LLVM-exception — the same license as LLVM, so this code can
live inside LLVM and MLIR without friction.
