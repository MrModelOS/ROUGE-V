# rouge-cuda — CUDA-совместимый слой ROUGE

Зародыш слоя совместимости с CUDA (см. `docs/03-software-architecture.md`, раздел «Рантайм и CUDA-совместимость»; архитектурное решение — `decisions/0005-cuda-compat`).

**Что это:** перехватчик CUDA Driver API и Runtime API (подход ZLUDA), который принимает PTX-ядра и исполняет их на CPU-эмуляторе. На этом каркасе позже появится настоящий бэкенд: компилятор ROUGE (MLIR/LLVM) → ISA ROUGE → ускоритель Crimson.

**Зачем такой каркас:**
1. Зафиксировать контракт API (какие CUDA-символы мы обещаем — их ровно столько, сколько реализовано).
2. Проверить «логику перехвата» end-to-end без NVIDIA-железа: скомпилированный против shim-заголовков код с PTX-ядрами реально исполняется и даёт верные результаты.
3. Дать точку интеграции: `rouge::load_real_symbol("cuMemAlloc")` — если в системе есть настоящая libcuda/libcudart, наш слой **пробрасывает вызовы в неё** (режим interposition), если нет — работает CPU-эмуляция. Это и есть поведение совместимого слоя в продакшене.

## Сборка и тесты

```bash
cd software/rouge-cuda
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Ожидаемый результат:

```
[ROUGE-CUDA compat layer] CPU-emulation backend active
PASS: driver-API vector add (N=4096) on CPU-emulated ROUGE device
PASS: driver-API block_reduce (shared memory + bar.sync) on CPU-emulated ROUGE device (16 blocks x 256 threads)
PASS: driver-API atomic_reduce (atom.add.f32/u32, red.add.f32/u32) on CPU-emulated ROUGE device
PASS: driver-API fp16_reduce (f16/bf16 cvt, fma, shared + bar.sync) on CPU-emulated ROUGE device
PASS: runtime-API memory copy pipeline
PASS: cuInit is exported for LD_PRELOAD interposition
```

## Структура

```
include/rouge/
  cuda.h               ← CUDA Driver API (подмножество): cuInit…cuLaunchKernel
  cuda_runtime_api.h   ← CUDA Runtime API (подмножество): cudaMalloc…cudaStreamDestroy
src/
  driver_api.cpp       ← реализация Driver API + форвардинг в реальную libcuda
  runtime_api.cpp      ← реализация Runtime API на базе Driver API
  forward.cpp          ← dlsym-проброс в настоящие библиотеки CUDA (interposition)
  ptx.h  ptx.cpp       ← парсер PTX (подмножество) + интерпретатор ядер на CPU
tests/
  test_driver_vadd.cpp         ← векторное сложение через Driver API и PTX-ядро
  test_driver_block_reduce.cpp ← shared memory + bar.sync (блок-редукция)
  test_driver_atomic_reduce.cpp← атомики atom.add/red.add (тот же PTX, что в AOT)
  test_driver_fp16_reduce.cpp  ← FP16/BF16 (тот же PTX, что в AOT)
  test_runtime_api.cpp         ← cudaMalloc/cudaMemcpy/streams через Runtime API
  test_interposer.cpp          ← проверка, что символы CUDA экспортируются для LD_PRELOAD
  embed_ptx.cmake              ← встраивает канонические .ptx из rouge-compiler/tests/kernels
```

Канонические ядра (`vadd`, `block_reduce`, `atomic_reduce`, `fp16_reduce`) хранятся в одном месте — `rouge-compiler/tests/kernels/*.ptx` — и исполняются байт-в-байт в обоих путях (AOT и интерпретатор).

## Что умеет PTX-интерпретатор (подмножество v0)

Парсинг: `.version/.target/.address_size`, заголовки `.visible .entry`, сигнатуры `.param`, метки, комментарии `//`, `.reg .b16` (16-битные регистры).

Инструкции: `ld.param.{u64,u32}`, `mov.{u32,u64,f32,u16,f16,bf16}`, `cvta.*` (global: identity,
адреса = host-адреса; shared: база scratchpad блока + смещение), `mul.wide.u32`,
`add/sub.{s64,u64,u32,s32}` (в т.ч. с литералами), `add/sub/mul/div.f32`,
`add/sub/mul/div.f16`, `add/sub/mul.rn.bf16`, `fma.rn.{f16,bf16}`, `neg.{f16,bf16}`,
`cvt.*` (включая `cvt.rn.{f16,bf16}.f32`, `cvt.f32.{f16,bf16}`, `cvt.{f16,bf16}↔int`),
`setp.{ge,gt,lt,le,eq,ne}.u32`, `bra`/`bra.uni` (+ предикаты `@p` / `@!p`),
`ld/st.global.{f32,u32,u64,b32,b64,f16,bf16,b16}` (+ `nc.`),
`.shared` + `ld/st.shared.{f32,u32,u64,b32,b64,f16,bf16,b16}` (в т.ч. адрес по символу `[sym]`),
`cvta.to.shared`, `atom.add`/`red.add` (`u32/s32/u64/s64/f32`, `monotonic`), `bar.sync`, `ret`, `exit`.

### Синхронизация: CTA-секвенсер (SIMT)

Потоки блока исполняются **в ногу**: все живые потоки CTA шагают по одной
инструкции за раунд; когда каждый живой поток упёрся в `bar.sync` — барьер
отпускается (Threads, вышедшие раньше, не считаются). Это честно моделирует
видимость записи в shared-память «до» `bar.sync` для чтений «после» — тот же
кернел `block_reduce.ptx` даёт одинаковые суммы в интерпретаторе и в AOT-пути
`rouge-compiler` (один PTX-файл, два пути исполнения).

Спецрегистры: `%tid.x/y/z`, `%ctaid.x/y/z`, `%ntid.x/y/z`, `%nctaid.x/y/z`.

Расширение набора — вопрос добавления веток в `execute_kernel`; дорожная карта опкодов ведётся по мере появления реальных моделей.

## Где это в общей картине

Слой CUDA-совместимости — вспомогательный, а не главный путь. Основной путь пользователя — открытые стандарты (Triton / SYCL / PyTorch). CUDA-compat нужен для **переноса легаси**: существующие приложения запускаются у нас «как есть», пока проект мигрирует их на нативный путь. При этом юридическая рамка жёсткая (ADR-0005): только открытые реализации по публичной документации, никакого копирования и реверса проприетарного кода NVIDIA.
