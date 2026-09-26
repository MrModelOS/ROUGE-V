# rouge-ptx

Shared PTX front-end for ROUGE-V: a parser for the PTX assembly produced by
`nvcc`, plus a reference interpreter that executes PTX on the host.

Both execution paths in this repository — the AOT translator in
[`rouge-compiler`](../rouge-compiler/) and the CUDA-compat layer in
[`rouge-cuda`](../rouge-cuda/) — are built on this parser. That is deliberate:
one grammar, one set of rules for registers, parameters, `.shared` layout and
labels, so the two paths cannot drift apart.

## What it provides

```c++
std::unique_ptr<rouge::PtxProgram> parse_ptx(const std::string& text,
                                             std::string* error);

bool execute_kernel(const rouge::PtxProgram& prog, int fnIndex,
                    const std::vector<uint8_t>& paramsBlob,
                    unsigned gx, unsigned gy, unsigned gz,
                    unsigned bx, unsigned by, unsigned bz,
                    std::string* error);
```

`parse_ptx` returns a `PtxProgram` holding functions with their parameters,
register widths (`.b16` / `.b32` / `.b64` / `.pred`), laid-out `.shared`
variables, and the instruction list with labels resolved. Parse errors are
returned through `error` with a line number.

`execute_kernel` runs a kernel over the whole grid on the host, allocating a
per-block shared-memory scratchpad and stepping the threads of each block in
lockstep, so `bar.sync` and shared-memory visibility match CUDA semantics.

## Notes

- The interpreter is a **correctness reference**, not a performance path. The
  AOT path exists because instruction-by-instruction execution has a ceiling
  that no amount of optimisation lifts.
- Instructions outside the supported subset fail with a message naming the
  instruction. They are not approximated.
- This is a header-only interface (`ptx.h`) with a single implementation file
  (`ptx.cpp`), no external dependencies.

## License

Apache 2.0 WITH LLVM-exception.
