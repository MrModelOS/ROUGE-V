# rouge-compiler — AOT-компилятор ROUGE-V

Основной путь выполнения CUDA-кода: **перехват на уровне IR → нативный
бинарник без эмуляции инструкций**. Архитектура: `docs/06-compiler-architecture.md`,
архитектура: [docs/06-compiler-architecture.md](../../docs/06-compiler-architecture.md).

## Что внутри

```
PTX (текстовый ассемблер CUDA)
   │  rouge-ptx (общий парсер: инструкции, параметры, типы регистров)
   ▼
ptx2ir ──► LLVM IR (define void @kernel(... %argN, ptr %launch))
   │
   ├─► clang -c          → нативный объект (x86/ARM/…)
   ├─► clang --target=riscv64-unknown-elf -march=rv64gcv -c → RISC-V + Vector
   └─► (план) MLIR: GPU → Vector/Linalg → ROUGE dialekt, tiling/coalescing
```

Кернел получает параметры как `i64/i32 %arg0..`, а CUDA-спецрегистры
(`%tid.x`, `%ctaid.x`, …) — из дескриптора запуска `ptr %launch`, который
заполняет хост-рантайм:

```
смещение 0..44: 12×i32 — tid/ctaid/ntid/nctaid × xyz
смещение 48   : i64     — база scratchpad блока (.shared)
смещение 56   : i64     — хендл барьера блока (bar.sync → __rouge_syncthreads)
```

`bar.sync` транслируется в `call void @__rouge_syncthreads(ptr %launch)` —
реализация живёт в `runtime/rouge_runtime.cpp` (хост: `pthread_barrier`;
целевой RISC-V: кастомный барьер/`fence`).

## Сборка и тесты

```sh
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

| Тест | Проверяет |
|---|---|
| `compiler_ptx_to_llvm` | структурная корректность IR (типы, ветвления, память) |
| `compiler_ptx_to_llvm_shared` | структурно: раскладка `.shared` (смещения), scratchpad-база в launch, lowering `bar.sync` |
| `compiler_ptx_to_llvm_atomics_fp16` | структурно: `atom.add`/`red.add` → `atomicrmw`, half/bfloat lowering, экспорт `.shared`-метаданных |
| `aot_native_vadd` | **нативный AOT**: ptx2ir → clang -O2 → объект → драйвер → `c[i]==a[i]+b[i]` для grid 16×256, без интерпретатора (требует `clang`) |
| `aot_native_block_reduce` | **shared + `bar.sync` в AOT**: блок-редукция 16×256 (pthread-барьер), scratchpad из `__rouge_*_query` — без `kScratch` в драйвере |
| `aot_native_atomic_reduce` | **атомики в AOT**: `atom.add`/`red.add` как `atomicrmw` на реальных потоках (гонка на кэш-линиях) |
| `aot_native_fp16_reduce` | **FP16/BF16 в AOT**: half/bfloat, cvt, fma, f16-тайл в shared + `bar.sync` (бит-идентично интерпретатору) |
| `compiler_rvv_backend_*` | тот же IR собирается бэкендом RISC-V `rv64gcv` (3 ядра: vadd, atomic, fp16 — информационные) |
| `mlir_simt_access_report` | MLIR-контур: `rouge-opt --rouge-simt-access-report` (только с `-DROUGE_ENABLE_MLIR=ON` + MLIR) |

## Поддерживаемое подмножество PTX

| Группа | Инструкции |
|---|---|
| Параметры | `ld.param.{u32,u64}` (+s/b) |
| Пересылки | `mov.{u32,u64,f32,u16,f16,bf16}`, `cvta.to.global.*`, `cvta.to.shared.*` |
| Арифметика | `add/sub/mul.{u32,s64,…}`, `add/sub/mul/div.f32`, `mul.wide.{u,s}32`, `add/sub/mul/div.f16`, `add/sub/mul.rn.bf16`, `fma.rn.{f16,bf16}` |
| Преобразования | `cvt.{u64.u32,u32.u64,s64.s32,s32.s64}`, `cvt.{f32↔u32/s32}`, `cvt.{f32↔f16, f32↔bf16, f16↔bf16, f16/bf16↔int}` (`rn`-квалификатор) |
| Сравнения | `setp.{ge,gt,lt,le,eq,ne}.u32` |
| Предикаты/ветвления | `@p bra`, `bra`, `ret`, `exit` |
| Память global | `ld.global.{f32,u32,u64,f16,bf16,b16}` (+nc), `st.global.{f32,u32,u64,f16,bf16,b16}` |
| Память shared | `.shared` (раскладка, `ld/st.shared.{f32,u32,u64,f16,bf16,b16}`, `[sym±const]`), scratchpad на launch-дескрипторе + экспорт `@__rouge_*_query` |
| Синхронизация | `bar.sync [N[, count]]` → `__rouge_syncthreads` |
| Атомики | `atom.add.{u32,s32,u64,s64,f32}` (→ `atomicrmw`, возвращает старое), `red.add.*` (без возврата), `monotonic` |

Неподдерживаемое (warp-shuffle, `atom.exch`/`cas`/`min/max`, `.f16x2`/`.bf16x2`, тензорные инструкции)
транслятор **отклоняет с ошибкой** — честный отказ, а не тихий мискомпил.

## Почему alloca-стиль

Каждый PTX-регистр — `alloca` в entry-блоке. Это всегда корректно (переживает
переопределения регистров и контрольный поток) и отлично оптимизируется LLVM
(mem2reg). На скалярном уровне уже работают `__shared__` → scratchpad на
launch-дескрипторе и `bar.sync` → рантайм-барьер. Следующий этап — MLIR-контур,
где скалярный SIMT-код векторизуется (Warp → RVV-VLEN), а барьеры лягут на
аппаратную синхронизацию RISC-V.

## План расширения

1. ~~атомики (`atom.*`/`red.*`) и FP16/BF16~~ ✅ (скалярный путь; RVV-векторизация — следующий шаг);
2. MLIR: GPU-диалект → Vector, SIMT→RVV даунсэмплинг (контур стартовал: `mlir/` + `rouge-simt-access-report`);
3. Hardware-Aware Passes: tiling, memory coalescing, double buffering;
4. Backend-расширение `rv64gcv_xrouge` (matmul.mma) и рантайм с Command Queue.