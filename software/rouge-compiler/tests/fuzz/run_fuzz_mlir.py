#!/usr/bin/env python3
"""ROUGE-V: the differential fuzzer for the MLIR contour.

For each seed:
  1. generate a random PTX kernel (gen_kernel);
  2. translate it with OUR PTX front-end (rouge-ptx2mlir) into a module of
     the `rouge` dialect;
  3. lower it (rouge-opt --rouge-to-scf) and through the standard
     MLIR -> LLVM IR -> binary chain;
  4. run the binary on the CPU;
  5. compare the printed output with the PTX interpreter's, element by
     element.

The interpreter is the REFERENCE: it is a completely independent
implementation (the text one), so a mismatch means that one of the two is
wrong. The same kernel goes through two paths that share nothing but the
PTX text.

This is the CI component that catches the "silent" bugs — the ones that
compile without errors and compute the wrong thing.

usage: run_fuzz_mlir.py --ptx2mlir <exe> --rouge-opt <exe> --gen <exe>
                        --interp <exe> --mlir-opt <exe> --mlir-translate <exe>
                        --clang <exe> [--runtime <print_i32.o>] [--iters N]
                        [--work DIR] [--cases DIR]
"""

import argparse
import os
import shutil
import struct
import subprocess
import sys

N = 64          # the elements of the input
K_OUT = 4       # the outputs per thread
OUT = N * K_OUT
SEED = 12345    # the seed the interpreter passes to the kernel


def run(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def harness_module(values):
    """The MLIR main: fill the input, launch the kernel, print the output.

    The input values are constants: the kernel must see exactly the bytes
    the interpreter saw.
    """
    lines = []
    # The declaration comes first: a call must see the symbol.
    lines.append("  func.func private @print_i32(i32)")
    lines.append("  func.func @main() {")
    lines.append(f"    %out = memref.alloc() : memref<{OUT}xi32>")
    lines.append(f"    %in = memref.alloc() : memref<{N}xi32>")
    lines.append("    %c0 = arith.constant 0 : index")
    lines.append("    %c1 = arith.constant 1 : index")
    lines.append(f"    %c{OUT} = arith.constant {OUT} : index")
    lines.append(f"    %n = arith.constant {N} : i32")
    lines.append(f"    %seed = arith.constant {SEED} : i32")
    for i, v in enumerate(values):
        lines.append(f"    %v{i} = arith.constant {v} : i32")
        lines.append(f"    %k{i} = arith.constant {i} : index")
        lines.append(f"    memref.store %v{i}, %in[%k{i}] : memref<{N}xi32>")
    lines.append(f"    func.call @fuzz_kernel_host(%out, %in, %n, %seed) : "
                 f"(memref<{OUT}xi32>, memref<{N}xi32>, i32, i32) -> ()")
    lines.append(f"    scf.for %i = %c0 to %c{OUT} step %c1 {{")
    lines.append(f"      %v = memref.load %out[%i] : memref<{OUT}xi32>")
    lines.append("      func.call @print_i32(%v) : (i32) -> ()")
    lines.append("    }")
    lines.append(f"    memref.dealloc %out : memref<{OUT}xi32>")
    lines.append(f"    memref.dealloc %in : memref<{N}xi32>")
    lines.append("    return")
    lines.append("  }")
    return "\n".join(lines)



def check_case(args, kern, in_bin, work):
    """One kernel through both paths. Returns (ok, detail).

    `detail` is the first difference, for the log: the saved case is only
    useful if you can see what it was.
    """
    ref_bin = os.path.join(work, "ref.bin")
    dialect_mlir = os.path.join(work, "kernel_dialect.mlir")
    harness_mlir = os.path.join(work, "kernel_harness.mlir")
    lowered_mlir = os.path.join(work, "kernel_lowered.mlir")
    ll = os.path.join(work, "kernel.ll")
    exe = os.path.join(work, "kernel.bin")

    # The reference: our PTX interpreter.
    r = run([args.interp, kern, in_bin, ref_bin, str(N)])
    if r.returncode != 0:
        return False, "the interpreter failed: %s" % r.stderr.strip()
    ref = open(ref_bin, "rb").read()
    ref_values = list(struct.unpack("<%di" % (len(ref) // 4), ref))

    # The front-end: PTX -> the rouge dialect.
    r = run([args.ptx2mlir, kern,
             "--shape", "0=%dxi32" % OUT, "--shape", "1=%dxi32" % N,
             "--grid", "1", "--block", str(N), "-o", dialect_mlir])
    if r.returncode != 0:
        return False, "the front-end refused the kernel: %s" % r.stderr.strip()

    # The harness: the main that fills the input and prints the output.
    values = list(struct.unpack("<%di" % N, open(in_bin, "rb").read()))
    text = open(dialect_mlir).read()
    # The module ends with the closing brace of the module: the harness goes
    # right before it.
    idx = text.rstrip().rfind("}")
    harness = text[:idx] + harness_module(values) + "\n}\n"
    open(harness_mlir, "w").write(harness)

    # The lowering: the SIMT structure becomes loops.
    r = run([args.rouge_opt, "--rouge-to-scf", harness_mlir, "-o", lowered_mlir])
    if r.returncode != 0:
        return False, "the lowering refused the kernel: %s" % r.stderr.strip()

    # MLIR -> LLVM IR -> binary.
    r = run([args.mlir_opt,
             "--convert-scf-to-cf", "--convert-cf-to-llvm",
             "--convert-arith-to-llvm", "--convert-index-to-llvm",
             "--expand-strided-metadata", "--finalize-memref-to-llvm",
             "--convert-math-to-llvm", "--convert-vector-to-llvm",
             "--convert-func-to-llvm", "--reconcile-unrealized-casts",
             lowered_mlir, "-o", os.path.join(work, "tmp.mlir")])
    if r.returncode != 0:
        return False, "mlir-opt failed: %s" % r.stderr.strip()
    r = run([args.mlir_translate, "--mlir-to-llvmir",
             os.path.join(work, "tmp.mlir"), "-o", ll])
    if r.returncode != 0:
        return False, "mlir-translate failed: %s" % r.stderr.strip()
    r = run([args.clang, "-O2", ll, args.runtime, "-o", exe])
    if r.returncode != 0:
        return False, "clang failed: %s" % r.stderr.strip()

    # The run. The exit code is not the point: the lowered main() does not
    # necessarily return 0 (the same note as in the x86 semantic check). The
    # PRINTED values are.
    r = run([exe])
    got = [int(line) for line in r.stdout.split() if line.strip()]
    if not got:
        return False, "the program printed nothing: %s" % r.stderr.strip()

    if got != ref_values:
        for i in range(min(len(ref_values), len(got))):
            if ref_values[i] != got[i]:
                return False, ("first differing element at %d: ref=%d got=%d"
                               % (i, ref_values[i], got[i]))
        return False, ("the lengths differ: ref=%d got=%d"
                       % (len(ref_values), len(got)))
    return True, ""


def replay(args):
    """Re-run every saved case. They must all still pass."""
    if not os.path.isdir(args.replay):
        print("replay: no directory %s — nothing to do" % args.replay)
        return 0
    names = sorted(n for n in os.listdir(args.replay)
                   if os.path.isdir(os.path.join(args.replay, n)))
    failures = 0
    for name in names:
        case = os.path.join(args.replay, name)
        kern = os.path.join(case, "kernel.ptx")
        in_bin = os.path.join(case, "in.bin")
        if not (os.path.isfile(kern) and os.path.isfile(in_bin)):
            continue
        work = os.path.join(args.work, name)
        os.makedirs(work, exist_ok=True)
        ok, detail = check_case(args, kern, in_bin, work)
        if ok:
            print("%s: ok" % name)
        else:
            failures += 1
            print("%s: FAILED — %s" % (name, detail))
    print("replay: %d cases, %d failures" % (len(names), failures))
    return 1 if failures else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptx2mlir", required=True)
    ap.add_argument("--rouge-opt", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--interp", required=True)
    ap.add_argument("--mlir-opt", required=True)
    ap.add_argument("--mlir-translate", required=True)
    ap.add_argument("--clang", required=True)
    ap.add_argument("--runtime", required=True,
                    help="the compiled print_i32 runtime")
    ap.add_argument("--iters", type=int, default=16)
    ap.add_argument("--template", default="chain",
                    choices=["chain", "phases", "demoted", "atomic"],
                    help="the kernel shape: 'chain' — the per-thread "
                         "arithmetic chain; 'phases' — shared memory + a "
                         "barrier between two phases")
    ap.add_argument("--work", default="fuzz-work-mlir")
    ap.add_argument("--cases", default="cases")
    ap.add_argument("--replay", default="",
                    help="re-run the saved cases in this directory instead of "
                         "generating new kernels")
    ap.add_argument("--ops", default="",
                    help="the operation chain (the generator's --ops mode); the "
                         "default is the generator's own random choice, which "
                         "may contain the operations the front-end does not "
                         "support yet (prmt, bfi, bfe) — those are REFUSED "
                         "loudly, and a refusal is a failure of this loop")
    args = ap.parse_args()

    os.makedirs(args.work, exist_ok=True)

    if args.replay:
        # Every saved case is a kernel that once found a bug. Re-running it
        # is the regression test: the same kernel, the same input, the same
        # comparison. A case that stops passing is a regression, and a case
        # that starts REFUSING is one too (the front-end must not lose a
        # feature that used to work).
        return replay(args)

    kern = os.path.join(args.work, "kernel.ptx")
    in_bin = os.path.join(args.work, "in.bin")

    failures = 0
    for seed in range(1, args.iters + 1):
        gen_cmd = [args.gen, "--seed", str(seed), "--out", kern, "--input", in_bin,
                   "--meta", os.path.join(args.work, "meta.txt"),
                   "--template", args.template]
        if args.ops:
            gen_cmd += ["--ops", args.ops]
        r = run(gen_cmd)
        if r.returncode != 0:
            print("seed %d: the generator failed: %s" % (seed, r.stderr.strip()))
            failures += 1
            continue

        ok, detail = check_case(args, kern, in_bin, args.work)
        if ok:
            if seed % 8 == 0:
                print("seed %d: ok" % seed)
            continue
        failures += 1
        case = os.path.join(args.cases, "mlir_seed_%d" % seed)
        os.makedirs(case, exist_ok=True)
        shutil.copy(kern, os.path.join(case, "kernel.ptx"))
        shutil.copy(in_bin, os.path.join(case, "in.bin"))
        shutil.copy(os.path.join(args.work, "kernel_harness.mlir"),
                    os.path.join(case, "kernel_harness.mlir"))
        print("seed %d: MISMATCH — saved to %s" % (seed, case))
        print("   " + detail)

    print("fuzz (MLIR): %d iterations, %d failures" % (args.iters, failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
