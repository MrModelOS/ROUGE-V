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
смещение 64   : i64     — база обменной области варпа (mma/ldmatrix/stmatrix;
                          NULL, если кернел без варп-коллективов)
```

`bar.sync` транслируется в `call void @__rouge_syncthreads(ptr %launch)` —
реализация живёт в `runtime/rouge_runtime.cpp` (хост: `pthread_barrier`;
целевой RISC-V: кастомный барьер/`fence`).

Обменная область варпа: 64 байта на поток блока (`rouge_exchange_alloc(n)`),
слот `+0` — i64 адрес строки (ldmatrix/stmatrix), `+8` — до 14×i32 слов
фрагментов (mma A/B). Коллектив публикует, барьер, читает слова соседей.

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
| `compiler_nvptx_backend` | IR собирается бэкендом NVIDIA `sm_75` (7 ядер, информационный) |
| `compiler_nvptx_gpu_exec` | ядра исполняются на настоящей NVIDIA GPU и сверяются побитово (12 конфигураций: 10 ядер + Tier-0 зонды чужого PTX, fp16/gemm по двум наборам данных + сетка 2×1; SKIP без sm_75+) |
| `compiler_nvptx_gpu_exec_atomic` | атомики `red`/`atom` исполняются на настоящей NVIDIA GPU (SKIP без sm_75+) |
| `compiler_nvcc_gpu_exec` | вендорный PTX (`nvcc real_nvcc.cu`) сквозь ptx2ir исполняется на настоящей NVIDIA GPU (SKIP без nvcc или sm_75+) |
| AMD-бэкенд — кросс-сборка | тот же IR с `addrspace(1)` (global) и `addrspace(5)` (shared) собирается в нативный объект `amdgcn-amd-amdhsa -mcpu=gfx1100` (5 ядер). Объект получен, но **ни разу не исполнялся** — AMD-видеокарты в проекте нет |
| NVIDIA-бэкенд — кросс-сборка и исполнение | тот же IR с `addrspace(1)` (global) и `addrspace(3)` (shared) собирается в нативный объект `nvptx64-nvidia-cuda -march=sm_75` (10 ядер) и **исполняется на GeForce MX450** — см. `compiler_nvptx_gpu_exec*` выше |
| `aot_native_cmp_signed` | **знаковые и беззнаковые сравнения**: `setp.lt.s32` не должен превращаться в беззнаковое сравнение (так и было) |
| `compiler_ptx_fuzz` | **дифференциальный фазер (хост)**: случайные PTX-ядра → наш ptx2ir → исполнение против интерпретатора, побитово |
| `compiler_ptx_fuzz_gpu` | **дифференциальный фазер (карта)**: то же, но арбитр — настоящая NVIDIA GPU (SKIP без sm_75+) |
| `mlir_simt_access_report` | MLIR-контур: `rouge-opt --rouge-simt-access-report` — отчёт по SIMT-паттернам (только с `-DROUGE_ENABLE_MLIR=ON` + MLIR; проверено на MLIR 23.1.3) |

Цель выбирается флагом `--target` (см. ниже). Строки `*_backend` — это проверка
**сборки**: бэкенд LLVM принял IR и выдал объект. Строки `*_gpu_exec` — проверка
**исполнения** на настоящей GeForce MX450. Ни те ни другие ничего не говорят о
производительности: мерялась только бит-идентичность результату хост-эталона.

## Дифференциальный фазер

Три независимых пути исполнения одного и того же ядра (текстовый интерпретатор
PTX, AOT-нижнинг, настоящая карта) и побитовое сравнение между ними давно были
в проекте — но проверяли пятнадцать ядер, написанных руками. Фазер закрывает
петлю: генератор выдаёт случайные PTX-ядра, `ptx2ir` их переводит, ответ карты
сравнивается с ответом интерпретатора **побайтово**. Расхождение
сохраняется в `tests/fuzz/cases/` и становится постоянным регрессионным тестом.

Что он нашёл (все находки — в компиляторе, не в генераторе):

1. **Знаковые сравнения компилировались как беззнаковые.** Вид `icmp` выбирался
   по УСЛОВИЮ (`lt` → `ult`), а знаковость из суффикса типа (`.s32` против
   `.u32`) игнорировалась: любое `setp.lt.s32` в любом ядре становилось
   беззнаковым. Интерпретатор был прав, карта следовала IR, и на отрицательном
   операнде они расходились. Это ровно тот класс дефекта, который не видно
   при чтении кода: компилируется без ошибок и считает не то.
2. **Предицированная инструк писала регистр всегда.** По PTX ISA (9.7.3) при
   ложном предикате регистр-назначение не пишется; нижник писал безусловно.
3. **Ветвление на несуществующую метку** давало IR со ссылкой на
   несуществующий блок, и замечал это только бэкенд. Теперь — громкий отказ.
4. **Типизированная ALU-инструкция с несовпадающей шириной назначения**
   (например `add.u32 %rd6, %rd6, 0` на 64-битном регистре) принималась
   молча и давала IR, который на хосте считал правдоподобно, а на карте — иначе.
   Ширина назначения теперь проверяется.
5. **`cvt.u64.u32` читал источник не той ширины**, если ширина регистра-источника
   не совпадала с типом в мнемонике, а `s64.u32` расширял знаковое значение
   через `zext` вместо `sext`. Оба случая исправлены.

Генератор намеренно не выдаёт ядра, читающие неинициализированные регистры
(на железе там мусор, в интерпретаторе ноль — сравнивать нечего) и избегает
деления на ноль (тоже неопределённый случай в PTX).

## Четыре цели сборки

`ptx2ir` принимает `--target <llvm-triple>`, и этот triple попадает в модуль как
`target triple = "..."`. Дальше IR собирается обычным `clang -c` с тем же
`--target` и соответствующим `-mcpu`/`-march`:

| `--target` | Платформа | Флаги бэкенда | Статус |
|---|---|---|---|
| *(по умолчанию)* `x86_64-pc-linux-gnu` | хост x86-64 | — | собирается и исполняется (`aot_native_*`) |
| `riscv64-unknown-elf` | RISC-V + Vector | `-march=rv64gcv` | собирается в объект |
| `amdgcn-amd-amdhsa` | AMD RDNA 3 | `-mcpu=gfx1100` | собирается в объект, не исполнялось |
| `nvptx64-nvidia-cuda` | NVIDIA | `-march=sm_75` | собирается в объект **и исполняется на GeForce MX450** (`compiler_nvptx_gpu_exec*`) |

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

## `rouge-cc` — drop-in замена nvcc

```sh
./build/rouge-compiler/rouge-cc -arch=sm_75 --target nvptx -c kernel.ptx -o kernel.o
./build/rouge-compiler/rouge-cc -arch=sm_75 -ptx kernel.cu -o kernel.ptx
```

Подмножество CLI nvcc (`-arch=sm_XX|native`, `-ptx`, `-c`, `-o`, `-I/-L/-D`,
`-O0…-O3`, `-g`, `-v`, `-n`) + расширение `--target host|riscv|amd|nvptx`.
`.cu` собирает только настоящий `nvcc` из PATH как чёрный ящик (`.cu → .ptx`),
своего CUDA-C фронтенда нет: без nvcc вход `.cu` — честная ошибка. `.ptx`
работает без nvcc. Линковка — только `--target host` (для device-целей `-c`
или `-ptx): device-линкера здесь нет.

## Поддерживаемое подмножество PTX

| Группа | Инструкции |
|---|---|
| Параметры | `ld.param.{u32,u64}` (+s/b) |
| Пересылки | `mov.{u32,u64,f32,u16,f16,bf16}`, `cvta.to.global.*`, `cvta.to.shared.*`, `cvta.local.*` (депо `.local`) |
| Арифметика | `add/sub/mul.{u32,s64,…}`, `add/sub/mul/div.f32`, `fma.rn.f32` (одиночное округление), `mul.wide.{u,s}32`, `add/sub/mul/div.f16`, `add/sub/mul.rn.bf16`, `fma.rn.{f16,bf16}` |
| Преобразования | `cvt.{u64.u32,u32.u64,s64.s32,s32.s64}`, `cvt.{f32↔u32/s32}`, `cvt.{f32↔f16, f32↔bf16, f16↔bf16, f16/bf16↔int}` (`rn`-квалификатор) |
| Сравнения | `setp.{ge,gt,lt,le,eq,ne}.u32` |
| Предикаты/ветвления | `@p bra`, `bra`, `ret`, `exit` |
| Память global | `ld.global.{f32,u32,u64,f16,bf16,b16}` (+nc), `st.global.{f32,u32,u64,f16,bf16,b16}`, generic `ld/st` без пробела, `.volatile` |
| Память shared | `.shared` + динамический `.shared` (unsized `.extern`), `ld/st.shared.*`, `[sym±const]`, scratchpad на launch-дескрипторе + экспорт `@__rouge_*_query` |
| Память local | `.local`-депо спиллов (`cvta.local`, `mov` адреса, generic доступы через `%SP`) |
| Синхронизация | `bar.sync` / `barrier.sync` / `bar.warp.sync` → `__rouge_syncthreads` (хост) / `llvm.nvvm.barrier0` (NVPTX) |
| Варп-шаффлы и голосования | `shfl.sync.*` + легаси `shfl.*`, включая предикатную форму `d\|p` (NVPTX: NVVM-интринсик + предикат из `%laneid`; хост: скаляр); `vote.sync.{any,all,uni}` (NVPTX: `llvm.nvvm.vote.*.sync`); `activemask.b32` (NVPTX: `llvm.nvvm.activemask`) |
| Атомики | `atom.add.{u32,s32,u64,s64,f32}` (→ `atomicrmw`, возвращает старое), `red.add.*` (без возврата), `monotonic` |

| Варп-коллективы матричные | `mma.sync.aligned` (хост: обмен фрагментами; NVPTX: железная инструкция через inline asm) и `ldmatrix`/`stmatrix.sync.aligned.m8n8.x{1,2,4}[.trans].b16` (хост: обмен адресами; NVPTX: железная инструкция). NVVM-интринсиков этих форм в LLVM нет (проверено пробой) — только inline asm. Архитектурные пороги честно: ldmatrix с sm_75 (исполнено на MX450), mma.sync m16n8k8/k16 с sm_80 (только сборка ptxas — тензорных ядер на TU117 нет), stmatrix с sm_90 (только сборка). |

Неподдерживаемое (`wmma.*`, `movmatrix`, split-фазные
`arrive`/`red`/`mbarrier.*`, `%clock64` на хосте, `mma`/`ldmatrix`/`stmatrix`
на NVPTX)
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
2. MLIR: GPU-диалект → Vector, SIMT→RVV даунсэмплинг (контур **собран и проверен**
   на MLIR 23.1.3: `mlir/` + `rouge-simt-access-report`; следующий шаг —
   GPU→Vector lowering);
3. Hardware-Aware Passes: tiling, memory coalescing, double buffering;
4. Backend-расширение `rv64gcv_xrouge` (matmul.mma) и рантайм с Command Queue.