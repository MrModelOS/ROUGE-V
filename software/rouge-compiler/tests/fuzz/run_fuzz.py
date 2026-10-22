#!/usr/bin/env python3
"""ROUGE-V differential fuzz loop.

For each seed:
  1. generate a random PTX kernel (gen_kernel);
  2. translate it with OUR ptx2ir (the NVPTX backend);
  3. lower it to PTX for the card with clang;
  4. run it on the real NVIDIA card (run_gpu) and on our PTX interpreter
     (run_interp);
  5. compare the outputs byte for byte.

A mismatch means our compiler and the hardware disagree — that is a bug in
one of them, and the case is saved under cases/ so it becomes a permanent
regression test instead of a flake.

Without an NVIDIA card the GPU side is skipped (the host side of the
fuzzer — interpreter vs AOT — is a separate loop, see --mode host).

usage: run_fuzz.py --ptx2ir <exe> --clang <exe> --gen <exe> --interp <exe>
                    --gpu <exe> [--mode gpu|host] [--iters N] [--work DIR]
"""

import argparse
import os
import shutil
import subprocess
import sys

K_OUT = 4
N = 64


def run(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptx2ir", required=True)
    ap.add_argument("--clang", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--interp", required=True)
    ap.add_argument("--gpu")
    ap.add_argument("--host-runner", help="run_host.cpp (the host half)")
    ap.add_argument("--runtime-lib", default="",
                    help="the host runtime (librouge-runtime.a): the kernel "
                         "calls into it (bar.sync -> __rouge_syncthreads)")
    ap.add_argument("--cxx", default="c++")
    ap.add_argument("--mode", default="gpu", choices=["gpu", "host"])
    ap.add_argument("--iters", type=int, default=16)
    ap.add_argument("--work", default="fuzz-work")
    ap.add_argument("--cases", default="cases")
    ap.add_argument("--sm", default="sm_75")
    ap.add_argument("--template", default="chain",
                    choices=["chain", "phases", "demoted", "atomic"],
                    help="the kernel shape (the generator's --template mode)")
    args = ap.parse_args()

    os.makedirs(args.work, exist_ok=True)
    kern = os.path.join(args.work, "kernel.ptx")
    ll = os.path.join(args.work, "kernel.ll")
    gpu_ptx = os.path.join(args.work, "kernel_gpu.ptx")
    host_obj = os.path.join(args.work, "kernel.o")
    in_bin = os.path.join(args.work, "in.bin")
    ref_bin = os.path.join(args.work, "ref.bin")
    got_bin = os.path.join(args.work, "got.bin")

    failures = 0
    skipped = 0
    for seed in range(1, args.iters + 1):
        r = run([args.gen, "--seed", str(seed), "--out", kern, "--input", in_bin,
                 "--meta", os.path.join(args.work, "meta.txt"),
                 "--template", args.template])
        if r.returncode != 0:
            print("seed %d: generator failed: %s" % (seed, r.stderr.strip()))
            failures += 1
            continue

        # Наш компилятор: PTX -> LLVM IR. Для GPU-режима — цель NVPTX, для
        # хостового — цели по умолчанию (x86_64): это РАЗНЫЕ .ll, и брать
        # NVPTX-модуль для host-компиляции нельзя (бэкенд не сможет выбрать
        # NVVM-интринсики).
        if args.mode == "gpu":
            r = run([args.ptx2ir, "--target", "nvptx64-nvidia-cuda", kern, ll])
        else:
            r = run([args.ptx2ir, kern, ll])
        if r.returncode != 0:
            print("seed %d: ptx2ir refused the kernel: %s"
                  % (seed, r.stderr.strip()))
            failures += 1
            continue

        # Эталон: наш интерпретатор.
        r = run([args.interp, kern, in_bin, ref_bin, str(N)])
        if r.returncode != 0:
            print("seed %d: interpreter failed: %s" % (seed, r.stderr.strip()))
            failures += 1
            continue

        if args.mode == "gpu":
            if not args.gpu:
                print("no --gpu given")
                return 2
            r = run([args.clang, "--target=nvptx64-nvidia-cuda", "-march=" + args.sm,
                     "-O2", "-S", ll, "-o", gpu_ptx])
            if r.returncode != 0:
                print("seed %d: clang refused the IR: %s" % (seed, r.stderr.strip()))
                failures += 1
                continue
            r = run([args.gpu, gpu_ptx, in_bin, got_bin, str(N)])
            if r.returncode != 0:
                if "NO-DEVICE" in r.stdout or "SKIP-OLD-ARCH" in r.stdout:
                    skipped += 1
                    if skipped == 1:
                        print("no NVIDIA card here — the GPU half of the fuzzer "
                              "is skipped, the host half still runs")
                    continue
                print("seed %d: GPU run failed: %s%s" % (seed, r.stdout.strip(),
                                                         r.stderr.strip()))
                failures += 1
                continue
        else:
            # Хостовая половина: AOT-объект + раннер собираются на каждый
            # seed (ядро линкуется в раннер статически).
            if not args.host_runner:
                print("no --host-runner given")
                return 2
            r = run([args.clang, "-O2", "-c", ll, "-o", host_obj])
            if r.returncode != 0:
                print("seed %d: clang refused the IR (host): %s"
                      % (seed, r.stderr.strip()))
                failures += 1
                continue
            host_exe = os.path.join(args.work, "run_host")
            link_cmd = [args.cxx, "-O2", args.host_runner, host_obj, "-o", host_exe]
            if args.runtime_lib:
                link_cmd.append(args.runtime_lib)
                link_cmd.append("-lpthread")
            r = run(link_cmd)
            if r.returncode != 0:
                print("seed %d: cannot link the host runner: %s"
                      % (seed, r.stderr.strip()))
                failures += 1
                continue
            r = run([host_exe, in_bin, got_bin, str(N)])
            if r.returncode != 0:
                print("seed %d: host run failed: %s" % (seed, r.stderr.strip()))
                failures += 1
                continue

        ref = open(ref_bin, "rb").read()
        got = open(got_bin, "rb").read()
        if ref != got:
            failures += 1
            case = os.path.join(args.cases, "seed_%d" % seed)
            os.makedirs(case, exist_ok=True)
            shutil.copy(kern, os.path.join(case, "kernel.ptx"))
            shutil.copy(in_bin, os.path.join(case, "in.bin"))
            print("seed %d: MISMATCH — saved to %s" % (seed, case))
            for i in range(0, min(len(ref), len(got)), 4):
                if ref[i:i + 4] != got[i:i + 4]:
                    print("   first differing word at %d: ref=%d got=%d"
                          % (i // 4, int.from_bytes(ref[i:i + 4], "little"),
                             int.from_bytes(got[i:i + 4], "little")))
                    break
        else:
            if seed % 8 == 0:
                print("seed %d: ok" % seed)

    print("fuzz: %d iterations, %d failures, %d skipped" %
          (args.iters, failures, skipped))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
