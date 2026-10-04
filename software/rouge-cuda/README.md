# rouge-cuda — CUDA-совместимый слой ROUGE

Слой совместимости с CUDA: перехват Driver/Runtime API и исполнение PTX-ядер. Архитектура проекта — в [docs/06](../docs/06-compiler-architecture.md).

**Что это:** перехватчик CUDA Driver API и Runtime API (подход ZLUDA), который принимает PTX-ядра и исполняет их на CPU-эмуляторе. Тот же каркас используется как fallback-путь и как стенд для сверки с AOT-компилятором.

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

## Imposter-режим: чужая программа без NVIDIA

Слой работает не только как LD_PRELOAD-перехватчик, но и как
подмена самой библиотеки. Бинарник, слинкованный против
настоящего `-lcuda` (чужая программа, исходников которой у
нас нет), запускается так:

```bash
mkdir imposter
cp build/librouge-cuda.so imposter/libcuda.so.1
LD_LIBRARY_PATH=$PWD/imposter ROUGE_CUDA_FORCE_EMU=1 ./foreign_app kernel.ptx
```

Как это устроено:
- лоадер ищет `libcuda.so.1` по LD_LIBRARY_PATH и находит наш
  файл; все вызовы CUDA Driver API попадают в нас;
- сентинел `rouge_cuda_emulation_sentinel` не даёт нам
  перепутать самих себя с «настоящей» libcuda при зондировании;
- `ROUGE_CUDA_FORCE_EMU=1` запрещает проброс в настоящий
  драйвер даже если он в системе есть — режим с нулём вызовов
  NVIDIA, пригодный для аудита;
- без этой переменной, наоборот, при наличии реальной libcuda
  вызовы пробрасываются в неё (режим interposition).

Бинарники, собранные против свежих заголовков CUDA 13, зовут
версионированные имена (`cuCtxCreate_v4`, `cuMemAlloc_v2`,
`cuMemcpyHtoD_v2`, …) — они реализованы и пробрасываются тем
же путём, что и базовые спеллинги.

## Запуск целой программы (Runtime API, сахар `<<<>>>`)

Драйверный API — только половина дела: настоящие программы собраны
через **Runtime API**, и компилятор NVIDIA реализует `<<<>>>`
не через `cudaLaunchKernel`, а через собственный протокол
регистрации. Протокол изучен по дизассемблированию объекта
`nvcc -arch=sm_75 -c`, а не по памяти:

```
.init_array → __sti____cudaRegisterAll():
    handle = __cudaRegisterFatBinary(...);   // возвращает handle
    __nv_cudaEntityRegisterCallback(handle); // код самой программы
    __cudaRegisterFatBinaryEnd(&handle);
колбэк программы, на ядро:
    __cudaRegisterFunction(handle, stub, mangled_name, ...);
запуск <<<1,64>>>:
    __cudaGetKernel(&handle, stub)          // один раз, под guard
    __cudaPushCallConfiguration(...) / __cudaPopCallConfiguration(...)
    __cudaLaunchKernel(handle, grid, block, args, shared, stream)
```

Мы перехватываем ровно эти точки: `stub → имя` — таблица
диспетчеризации, а PTX берётся из `ROUGE_CUDA_PTX` (разбор fatbin
следующий шаг и протокола не меняет).

Проверка `tests/programs/wave1_vadd.cu`: **один и тот же исходник**
собран дважды — `nvcc` против настоящего cudart и `g++` против
нашего шима, без единой ссылки на CUDA-библиотеку. Оба запуска
печатают `OK`; тест сравнивает выводы и валит прогон при расхождении.
GPU для этого не нужен вообще.

## Покрытие Runtime API (Волна 1)

Помимо запуска, настоящие программы читают свойства устройства и
пользуются pinned-памятью, стримами и асинхронными копиями. Всё это
реализовано и проверено программой `tests/programs/wave1_api.cu`
(тот же принцип: сборка `nvcc` + сборка против шима, сравнение выводов):

- `cudaSetDevice` / `cudaGetDevice` / `cudaGetDeviceCount` / `cudaDeviceReset`;
- `cudaGetDeviceProperties` — структура `rouge_cuda_device_prop` повторяет
  раскладку настоящего `cudaDeviceProp` поле в поле (93 поля); тест
  `device_prop_layout` сверяет `offsetof`/`sizeof` каждого поля с настоящим
  заголовком на этапе сборки. Значения честные: имя и UUID наши, память —
  физическая (sysconf), `major.minor` = 7.5 (референсный таргет), варп 32,
  часы — нули (измерять нечего, выдумывать нельзя);
- `cudaMallocHost` / `cudaHostAlloc` / `cudaFreeHost` (pinned = обычная куча:
  DMA гнать некуда, unified addressing = 1);
- `cudaMemcpyAsync` / `cudaMemset` / `cudaPeekAtLastError` (не сбрасывает
  sticky-ошибку, в отличие от Get);
- `cudaStreamCreateWithFlags` (флаги принимаются, бэкенд синхронный —
  это задокументировано, а не спрятано).

## Occupancy и атрибуты функций

Настоящие программы спрашивают у драйвера, сколько блоков влезет на SM,
и подбирают конфигурацию запуска. Мы считаем по тому же документированному
алгоритму из открытого `cuda_occupancy.h` (минимум лимитов: варпы, регистры,
блоки, shared) для референсного sm_75:

- `cudaFuncGetAttributes` / `cuFuncGetAttribute`: shared/local/const размеры
  и `.target` модуля — из распарсенного PTX; `numRegs` — max-live оценка
  (dataflow-анализ тела; настоящий аллокатор ptxas недоступен, его данные
  лежат в недокументированном `.nv.info` — это вне legal frame);
- `cudaOccupancyMaxActiveBlocksPerMultiprocessor[WithFlags]`,
  `cudaOccupancyMaxPotentialBlockSize`, `cudaOccupancyAvailableDynamicSMemPerBlock`
  и driver-варианты `cuOccupancy*`;
- `cudaDeviceGetAttribute` (честный набор: варп 32, блоки, SM, CC 7.5).

Архитектурные константы sm_75 сверены пробой с настоящим драйвером MX450,
а не взяты по памяти (так пойманы `maxThreadsPerMultiProcessor = 1024`,
а не 2048, и пять неверных значений в props). Невозможная конфигурация —
успех с нулём блоков, как у настоящего драйвера; чужие флаги — отказ.

Проверка — программа `tests/programs/wave1_occupancy.cu`: собирается `nvcc`
против настоящего cudart и `g++` против шима, occupancy-числа сравниваются
побайтово (нормализуются только имя устройства, minGridSize — топология
14 SM против 1 — и regs — точная аллокация против max-live оценки).

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

Парсинг: `.version/.target/.address_size`, заголовки `.visible .entry`, сигнатуры `.param`, метки (в т.ч. `$L__BB0_2` от nvcc 13 — `$` валиден, метка в любом месте строки, trailing-комментарий допустим), комментарии `//`, `.reg .b16` (16-битные регистры).

Инструкции: `ld.param.{u64,u32}`, `mov.{u32,u64,f32,u16,f16,bf16}`, `cvta.*` (global: identity,
адреса = host-адреса; shared: база scratchpad блока + смещение), `mul.wide.u32`,
`add/sub.{s64,u64,u32,s32}` (в т.ч. с литералами), `add/sub/mul/div/fma.rn.f32` (fused,
одно округление через `std::fma` — как железный FMA и `llvm.fma`),
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

Модульные `.global`/`.const`: backing store выделяет `cuModuleGetGlobal` (нулями,
как device memory); интерпретатор видит те же адреса через `mov`/`cvta`/прямые
`[sym]`, поэтому host-запись через возвращённый указатель видна последующим
запускам кернелов. Хранилище живёт до `cuModuleUnload`.

Расширение набора — вопрос добавления веток в `execute_kernel`; дорожная карта опкодов ведётся по мере появления реальных моделей.

## Где это в общей картине

Слой CUDA-совместимости — вспомогательный, а не главный путь. Основной путь пользователя — открытые стандарты (Triton / SYCL / PyTorch). CUDA-compat нужен для **переноса легаси**: существующие приложения запускаются у нас «как есть», пока проект мигрирует их на нативный путь. Юридическая рамка жёсткая: только открытые реализации по публичной документации, никакого копирования и реверса проприетарного кода NVIDIA.
