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
| `compiler_rvv_backend_*` | тот же IR собирается бэкендом RISC-V `rv64gcv` (информационные) |
| `compiler_amdgpu_backend` | IR собирается бэкендом AMD `gfx1100` (5 ядер, информационный; SKIP без ROCm) |
| `compiler_nvptx_backend` | IR собирается бэкендом NVIDIA `sm_75` (5 ядер, информационный) |
| AMD-бэкенд — кросс-сборка | тот же IR с `addrspace(1)` (global) и `addrspace(5)` (shared) собирается в нативный объект `amdgcn-amd-amdhsa -mcpu=gfx1100` (5 ядер). Объект получен, но **ни разу не исполнялся** — GPU в проекте нет |
| NVIDIA-бэкенд — кросс-сборка | тот же IR с `addrspace(1)` (global) и `addrspace(3)` (shared) собирается в нативный объект `nvptx64-nvidia-cuda -march=sm_75` (5 ядер). Собирается, но **ни разу не исполнялось** — GPU в проекте нет |
| `mlir_simt_access_report` | MLIR-контур: `rouge-opt --rouge-simt-access-report` (только с `-DROUGE_ENABLE_MLIR=ON` + MLIR) |

Цель выбирается флагом `--target` (см. ниже). Две строки GPU-бэкендов — это
проверка **сборки**: бэкенд LLVM принял IR и выдал объект. Исполнения на реальном
железе не было, поэтому строки ничего не говорят о производительности.

## Четыре цели сборки

`ptx2ir` принимает `--target <llvm-triple>`, и этот triple попадает в модуль как
`target triple = "..."`. Дальше IR собирается обычным `clang -c` с тем же
`--target` и соответствующим `-mcpu`/`-march`:

| `--target` | Платформа | Флаги бэкенда | Статус |
|---|---|---|---|
| *(по умолчанию)* `x86_64-pc-linux-gnu` | хост x86-64 | — | собирается и исполняется (`aot_native_*`) |
| `riscv64-unknown-elf` | RISC-V + Vector | `-march=rv64gcv` | собирается в объект |
| `amdgcn-amd-amdhsa` | AMD RDNA 3 | `-mcpu=gfx1100` | собирается в объект, не исполнялось |
| `nvptx64-nvidia-cuda` | NVIDIA | `-march=sm_75` | собирается в объект, не исполнялось |

```sh
./build/rouge-compiler/ptx2ir --target amdgcn-amd-amdhsa   kernel.ptx kernel-amd.ll
clang --target=amdgcn-amd-amdhsa   -mcpu=gfx1100 -c kernel-amd.ll -o kernel-amd.o
./build/rouge-compiler/ptx2ir --target nvptx64-nvidia-cuda kernel.ptx kernel-sm75.ll
clang --target=nvptx64-nvidia-cuda -march=sm_75   -c kernel-sm75.ll -o kernel-sm75.o
```

`--target` влияет не только на строку triple, но и на типы указателей в IR:
global — `addrspace(1)` у обоих GPU-бэкендов, shared — `addrspace(5)` у AMDGPU
и `addrspace(3)` у NVPTX, а для CPU остаётся generic `ptr`. Без этого бэкенд
печатает скалярные обращения вместо `global_load`/`global_store`. Подробности —
`docs/06-compiler-architecture.md`.

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