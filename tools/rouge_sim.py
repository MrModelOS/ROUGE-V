#!/usr/bin/env python3
"""
ROUGE-V ISA v0.1 — cycle-приблизительный симулятор тайлового GEMM.

Условная модель вычислительного тайла (оценка):
  - Scratchpad SRAM: тайлы A[Tm×Tk] + B[Tk×Tn] + C[Tm×Tn] живут в SRAM
  - Векторные lanes: VLEN=512 (элементов за вектор-инструкцию)
  - DMA ping-pong (double buffering): загрузка следующего тайла
    перекрывается с вычислением текущего
  - Пиковая производительность считается через lanes, DMA — через bandwidth

Оценка, не RTL. Цель — быстро прикидывать Tm/Tn/Tk и видеть
compute-bound vs memory-bound.
"""
from __future__ import annotations
import argparse
import math

# Ссылки:
#   Параметры условные; это калькулятор модели, а не спецификация продукта.
#   software/rouge-compiler/mlir/README.md — контур MLIR (SimtAccessReport)
# Примеры:
#   python3 tools/rouge_sim.py --m 128 --n 128 --k 128
#   python3 tools/rouge_sim.py --m 512 --n 512 --k 512 --tm 64 --tn 64 --tk 32 --bw 128
#   python3 tools/rouge_sim.py --m 64 --n 64 --k 64 --tm 16 --tn 16 --tk 16 --dtype-bytes 2
# Ограничения модели:
#   - не моделирует конфликты банков SRAM, NoC, HBM latency
#   - DMA считается как bytes/BW, без учёта burst/stride
#   - overlap — идеальный double buffering без пузырьков


# --- Дефолты железа (можно переопределить флагами) ---
DEFAULT_VLEN = 512          # элементов / вектор-инструкцию
DEFAULT_BW = 64             # байт / цикл ( ~64 ГБ/с при 1 ГГц)
DEFAULT_FREQ_GHZ = 1.0      # ГГц, для оценки TOPS
DEFAULT_DTYPE_BYTES = 1     # FP8 (для FP16 ставь 2)
DEFAULT_SRAM_KB = 256       # КБ на тайл (инфо, не ограничивает пока)
DEFAULT_TM = 32
DEFAULT_TN = 32
DEFAULT_TK = 32


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def simulate(m: int, n: int, k: int, tm: int, tn: int, tk: int,
             vlen: int, bw: int, freq_ghz: float, dtype_bytes: int):
    # --- тайлы ---
    mt = ceil_div(m, tm)
    nt = ceil_div(n, tn)
    kt = ceil_div(k, tk)
    num_tiles = mt * nt * kt
    total_flops = 2 * m * n * k  # FMA = 2 FLOPS

    # FLOPS на один тайл (последние тайлы могут быть меньше — берём среднее)
    flops_per_tile_avg = total_flops / num_tiles if num_tiles else 0
    # Альтернативно пиковый тайл:
    flops_per_full_tile = 2 * tm * tn * tk

    # --- циклы на вычисление одного тайла ---
    # 1 вектор-инструкция = VLEN FMA (т.е. VLEN умножений + сложений)
    # считаем: элементов MAC на тайл = tm*tn*tk, инструкций = ceil(tm*tn*tk / vlen)
    mac_per_full_tile = tm * tn * tk
    compute_cycles_per_tile = ceil_div(mac_per_full_tile, vlen)
    # если нужен учёт 2 FLOPS/FMA — не меняет циклы, но влияет на TOPS

    # --- циклы на DMA одного тайла ---
    # грузим A[tm×tk] + B[tk×tn], C — в SRAM (пишем обратно 1 раз на mt*nt тайлов)
    bytes_a = tm * tk * dtype_bytes
    bytes_b = tk * tn * dtype_bytes
    bytes_per_tile = bytes_a + bytes_b
    # запись C: Tm*Tn на каждую пару (i,j) один раз (k-тайлы аккумулируют в SRAM)
    # усредняем на тайл: bytes_c_write_per_tile = tm*tn*dtype_bytes / kt
    bytes_c_per_tile = (tm * tn * dtype_bytes) / kt if kt else 0
    bytes_per_tile_total = bytes_per_tile + bytes_c_per_tile
    dma_cycles_per_tile = math.ceil(bytes_per_tile_total / bw)

    # --- overlap double buffering ---
    if num_tiles == 0:
        total_cycles = 0
    elif num_tiles == 1:
        total_cycles = dma_cycles_per_tile + compute_cycles_per_tile
    else:
        # первый DMA не перекрыт, далее max, последний compute не перекрыт
        total_cycles = dma_cycles_per_tile + (num_tiles - 1) * max(dma_cycles_per_tile, compute_cycles_per_tile) + compute_cycles_per_tile

    # --- производные метрики ---
    ideal_compute_cycles = math.ceil(total_flops / 2 / vlen)  # если бы без тайлинга
    utilization = ideal_compute_cycles / total_cycles if total_cycles else 0
    # TOPS при freq ГГц: TOPS = FLOPS / (cycles/freq) /1e12
    secs = total_cycles / (freq_ghz * 1e9) if freq_ghz else 0
    tops = (total_flops / secs / 1e12) if secs else 0
    peak_tops = (vlen * 2 * freq_ghz * 1e9) / 1e12  # VLEN FMA *2 * freq

    # SRAM footprint полного тайла
    sram_bytes = (tm * tk + tk * tn + tm * tn) * dtype_bytes
    sram_kb = sram_bytes / 1024

    bound = "COMPUTE" if compute_cycles_per_tile >= dma_cycles_per_tile else "MEMORY"

    return {
        "mt": mt, "nt": nt, "kt": kt, "num_tiles": num_tiles,
        "total_flops": total_flops,
        "flops_per_full_tile": flops_per_full_tile,
        "compute_cycles_per_tile": compute_cycles_per_tile,
        "dma_cycles_per_tile": dma_cycles_per_tile,
        "bytes_per_tile": int(bytes_per_tile_total),
        "sram_kb": sram_kb,
        "total_cycles": total_cycles,
        "total_secs_us": secs * 1e6,
        "tops": tops,
        "peak_tops": peak_tops,
        "utilization": utilization,
        "bound": bound,
        "ideal_compute_cycles": ideal_compute_cycles,
    }


def main():
    p = argparse.ArgumentParser(description="ROUGE-V GEMM cycle simulator v0.1")
    p.add_argument("--m", type=int, required=True, help="M")
    p.add_argument("--n", type=int, required=True, help="N")
    p.add_argument("--k", type=int, required=True, help="K")
    p.add_argument("--tm", type=int, default=DEFAULT_TM, help="Tm tile M")
    p.add_argument("--tn", type=int, default=DEFAULT_TN, help="Tn tile N")
    p.add_argument("--tk", type=int, default=DEFAULT_TK, help="Tk tile K")
    p.add_argument("--vlen", type=int, default=DEFAULT_VLEN, help="VLEN lanes")
    p.add_argument("--bw", type=int, default=DEFAULT_BW, help="DMA bytes/cycle")
    p.add_argument("--freq", type=float, default=DEFAULT_FREQ_GHZ, help="freq GHz")
    p.add_argument("--dtype-bytes", type=int, default=DEFAULT_DTYPE_BYTES, help="bytes per element")
    p.add_argument("--sram-kb", type=int, default=DEFAULT_SRAM_KB, help="SRAM KB (info)")
    args = p.parse_args()

    r = simulate(args.m, args.n, args.k, args.tm, args.tn, args.tk,
                 args.vlen, args.bw, args.freq, args.dtype_bytes)

    print(f"ROUGE-V GEMM simulator v0.1 — M={args.m} N={args.n} K={args.k}  tile Tm={args.tm} Tn={args.tn} Tk={args.tk}")
    print(f"  VLEN={args.vlen}  BW={args.bw} B/cycle  freq={args.freq} GHz  dtype={args.dtype_bytes}B  SRAM~{r['sram_kb']:.1f} KB/tile")
    print(f"  tiles: {r['mt']}×{r['nt']}×{r['kt']} = {r['num_tiles']}  (total FLOPS={r['total_flops']:,})")
    print(f"  per-tile: compute {r['compute_cycles_per_tile']} cyc  DMA {r['dma_cycles_per_tile']} cyc ({r['bytes_per_tile']} B)  → {r['bound']}-bound")
    print(f"  total: {r['total_cycles']:,} cycles  ({r['total_secs_us']:.2f} us @ {args.freq} GHz)")
    print(f"  TOPS: {r['tops']:.3f}  (peak {r['peak_tops']:.3f} TOPS, util {r['utilization']*100:.1f}%)")
    print(f"  ideal compute cycles (no tiling/DMA): {r['ideal_compute_cycles']:,}")
    if r['sram_kb'] > args.sram_kb:
        print(f"  [WARN] tile {r['sram_kb']:.1f} KB > SRAM {args.sram_kb} KB — урезай Tm/Tn/Tk")


if __name__ == "__main__":
    main()
