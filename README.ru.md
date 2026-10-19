<h1 align="center">ROUGE-V</h1>

<p align="center">Компилятор, который переводит CUDA PTX в нативный код для RISC-V.</p>

<p align="center">
<a href="https://github.com/MrModelOS/ROUGE-V/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/MrModelOS/ROUGE-V/ci.yml?style=flat-square&branch=main"></a>
<img alt="Тесты" src="https://img.shields.io/badge/tests-24%2F24%20passed-brightgreen?style=flat-square">
<img alt="Язык" src="https://img.shields.io/badge/language-C%2B%2B-blue?style=flat-square">
<img alt="Лицензия" src="https://img.shields.io/badge/license-Apache--2.0%20WITH%20LLVM--exception-blue?style=flat-square">
</p>

<p align="center">
<a href="README.md">English</a> ·
<a href="README.ru.md">Русский</a> ·
<a href="README.zh.md">中文</a>
</p>

---

`nvcc` не генерирует машинный код для GPU. Он генерирует **PTX** — текстовую
ассемблерную нотацию с опубликованной спецификацией. Всё, что умеет читать PTX,
может скомпилировать его под другую платформу.

ROUGE-V делает именно это: PTX → LLVM IR → нативный код, для RISC-V и его
векторного расширения `rv64gcv`. Это компилятор, а не эмулятор: переведённое
ядро компилируется, а не исполняется инструкция-за-инструкцией.

### Установка

```sh
git clone https://github.com/MrModelOS/ROUGE-V.git
cd ROUGE-V
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Нужны CMake ≥ 3.20, компилятор C++17 и `clang`. `nvcc` не обязателен — он
включает тест с реальным PTX от вендора.

### Использование

Без сборки — два бинарника без зависимостей, кроме системного libc:

```sh
curl -fsSL https://github.com/MrModelOS/ROUGE-V/releases/download/v0.1/rouge-v-0.1-linux-x86_64.tar.gz | tar xz -C ~/.local
export PATH="$HOME/.local/bin:$PATH"
ptx2ir kernel.ptx kernel.ll
```

Запуск на NVIDIA GPU (sm_75+, нужен только драйвер, тулкит на запуске не нужен):

```sh
rouge-run --target nvptx64-nvidia-cuda kernel.ptx --emit-ptx kernel_gpu.ptx
# дальше 3 точки EDIT в software/rouge-compiler/examples/run_on_gpu.c
# (путь к PTX, имя ядра, буферы/запуск) и сборка против libcuda
```

(Arch Linux: `packaging/PKGBUILD` для ручной сборки. `packaging/install.sh` делает то же самое.)

```sh
./build/rouge-compiler/ptx2ir \
    software/rouge-compiler/tests/kernels/vadd.ptx /tmp/vadd.ll

clang -O2 -c /tmp/vadd.ll -o /tmp/vadd.o                  # хост
clang --target=riscv64-unknown-elf -march=rv64gcv \
      -c /tmp/vadd.ll -o /tmp/vadd-rv64.o                 # RISC-V + Vector
```

Тройную цель выбирают на стороне транслятора флагом `--target` — это просто
префикс аргументов `ptx2ir`, остальная команда не меняется. Один и тот же PTX
собирается под четыре платформы:

```sh
./build/rouge-compiler/ptx2ir --target amdgcn-amd-amdhsa kernel.ptx kernel-amd.ll
clang --target=amdgcn-amd-amdhsa -mcpu=gfx1100 -c kernel-amd.ll -o kernel-amd.o
```

| `--target` | Платформа | Флаги бэкенда | Что на самом деле проверено |
|---|---|---|---|
| *(по умолчанию)* `x86_64-pc-linux-gnu` | хост x86-64 | — | собирается **и исполняется** — тесты `aot_native_*` |
| `riscv64-unknown-elf` | RISC-V + Vector | `-march=rv64gcv` | собирается в нативный объект — `ctest compiler_rvv_backend_*`; ядро без барьеров ещё и **выполняется** под `qemu-riscv64` — `ctest compiler_rv_exec` (по одному lane; ядра с `bar.sync` требуют настоящих потоков и не заявлены) |
| `amdgcn-amd-amdhsa` | AMD RDNA 3 | `-mcpu=gfx1100` | собирается в нативный объект — не исполнялось |
| `nvptx64-nvidia-cuda` | NVIDIA | `-march=sm_75` | собирается **и исполняется на настоящей видеокарте** — см. ниже |

Под все четыре тройки собираются пятнадцать канонических ядер из
`software/rouge-compiler/tests/kernels/`: `vadd`, `block_reduce`,
`atomic_reduce`, `fp16_reduce`, `fp8_cvt`, `gemm_tile`, `shfl_reduce`, `atom_cas`,
`shfl_pred`, `vote_pred`, `shfl_pred_all`, `mma_tile`, `mma_forms`, `mma_pipe`,
`ldmatrix_tile`.

### Исполнено на реальном железе

Ядра для NVIDIA не только собираются: `ctest compiler_nvptx_gpu_exec*`
загружает полученный PTX через CUDA Driver API (JIT на стороне драйвера) на
настоящей GeForce MX450 (sm_75) и сверяет каждый выход побитово с хост-эталоном:
все десять канонических ядер, два набора данных для `fp16_reduce`/`gemm_tile`
плюс сетка из двух блоков, проверяющая многоблочный путь. `compiler_nvcc_gpu_exec`
дополнительно гоняет настоящее вендорное ядро (`nvcc real_nvcc.cu`) насквозь. На
машинах без NVIDIA GPU sm_75+ тесты дают SKIP, а не провал.

`ldmatrix` — первый варп-коллектив матричного семейства, исполненный на
настоящем кремнии (sm_75: это трафик в shared-память, тензорные ядра не нужны).
`mma` и `stmatrix` собираются `ptxas` под sm_80/sm_90, но на этой карте
тензорных ядер нет, поэтому их **исполнение** остаётся только compile-verify.

Строка AMD остаётся утверждением о **сборке**: AMD-видеокарты в проекте нет.
Ни та ни другая строка ничего не говорят о производительности: мерялась только
бит-идентичность хост-эталону. Цель меняет адресные пространства в эмитимом IR —
именно поэтому эти бэкенды печатают `global_load`/`global_store`, а не скалярные
обращения; подробности в
[docs/06-compiler-architecture.md](docs/06-compiler-architecture.md).

```
$ ctest --test-dir build
...
 7/56 Test  #7: aot_native_block_reduce ..................   Passed    0.08 sec
 8/56 Test  #8: aot_native_atomic_reduce .................   Passed    0.07 sec
23/56 Test #23: compiler_foreign_ptx .....................   Passed    0.36 sec
33/56 Test #33: compiler_nvptx_gpu_exec ..................   Passed    7.73 sec
35/56 Test #35: compiler_nvcc_gpu_exec ...................   Passed    0.52 sec
100% tests passed out of 56
```

> [!NOTE]
> `compiler_nvcc_ptx` компилирует настоящий CUDA C штатным `nvcc` и прогоняет
> полученный PTX через весь конвейер. Тест пропускается, если `nvcc` не установлен.

> [!IMPORTANT]
> Инструкции вне поддерживаемого подмножества приводят к ошибке с указанием
> самой инструкции. Тихого мискомпилирования нет: неверный ответ хуже отказа.

### Компоненты

| Компонент | Что это |
|---|---|
| [`rouge-ptx`](software/rouge-ptx/) | Общий PTX-фронтенд: парсер и эталонный интерпретатор |
| [`rouge-compiler`](software/rouge-compiler/) | `ptx2ir`, AOT-транслятор; рантайм; контур MLIR |
| [`rouge-cuda`](software/rouge-cuda/) | Перехват CUDA Driver/Runtime API и запасной путь исполнения |
| [`rouge-numerics`](software/rouge-numerics/) | Своя открытая математика: GEMM (f32/f64/f16/bf16/FP8/комплекс), свёртка, attention, FFT, sparse, RNG, INT8-квантование — плюс шимы cuBLAS/cuDNN поверх своего кода |
| [`rouge-render`](software/rouge-render/) | Свой рендерер: path tracer, à-trous денойзер, темпоральный апскейлер — открытый путь там, где был бы DLSS/OptiX |

Оба пути исполнения построены на одном парсере. Это сделано намеренно: одна
грамматика, один набор правил для регистров, параметров, раскладки `.shared` и
меток — чтобы пути не расходились. Тесты строятся вокруг требования, что на
каждом ядре они дают **бит-в-бит одинаковый** результат.

### Инференс и графика — открытым способом

Две части стека, которые существуют, чтобы ничего закрытого не осталось:

- **Инференс** (`rouge-numerics`): INT8-квантование в трёх схемах (per-tensor,
  per-channel, асимметричное per-group с целочисленной нулевой точкой),
  **точный** целочисленный GEMM (накопление в int32), GQA/MQA-внимание со
  скользящим окном и FP8-KV-кэшем в нашем формате e4m3, и полный
  pre-norm трансформерный блок (RMSNorm → RoPE → внимание → SwiGLU-MLP), чьи
  проекции считаются нашим блочным GEMM. На тестовой машине блок даёт
  36.6 GFLOP/s; перенос проекций с наивной тройной петли на наш GEMM ускорил
  его **в 14 раз** (566 → 40.5 мс).
- **Графика** (`rouge-render`): path tracer (next-event estimation + MIS, чтобы
  эталон был несмещённым), à-trous вейвлет-денойзер и темпоральный апскейлер с
  джиттер-осознанной реконструкцией, репроекцией истории и variance clipping.
  Замеры против сходящегося эталона того же рендера (i5-1135G7, 64×64, эталон
  256 spp):

  | Шаг | Наш результат | База | Итог |
  |---|---|---|---|
  | Денойзер кадра в 4 отсчёта | PSNR 28.79, SSIM 0.927 | шум: 27.91 / 0.873 | +0.88 дБ, +0.054 SSIM |
  | Апскейл 2× (камера стоит) | PSNR 25.08, SSIM 0.816 | билинейный: 21.60 / 0.693 | +3.5 дБ, +0.12 SSIM |
  | Апскейл 2× (камера едет) | PSNR 25.25, SSIM 0.826 | билинейный: 21.60 / 0.693 | +3.7 дБ, +0.13 SSIM |

  Это **не DLSS**: у DLSS реконструкция нейросетевая (сеть закрыта), у нас
  классический конвейер (бикуб + репроекция + клиппинг). Смысл в том, что весь
  путь открыт, наш и измерим.

### Поддерживаемое подмножество PTX

| Группа | Инструкции |
|---|---|
| Параметры | `ld.param.{u32,u64}` и знаковые/битовые варианты |
| Пересылки | `mov.*`, `cvta.to.global.*`, `cvta.to.shared.*` |
| Арифметика | `add` `sub` `mul` `div` `rem`, `mad.lo` `mad.hi`, `mul.wide.*` |
| Побитовые / сдвиги | `and` `or` `xor` `not`, `shl`, `shr.{u,s}`, `bfi` `bfe` |
| Сравнения | `setp.{ge,gt,lt,le,eq,ne}.{u32,s32,f32}` |
| Выбор | `selp.*`, `slct.*`, предикаты `and` `or` `not` |
| Минимум / максимум / унарные | `min` `max.*`, `neg`, `abs`, `popc`, `clz`, `bfind.shiftamt` |
| Плавающая точка | арифметика `f16` / `bf16`, `fma.rn`, `sqrt`, `rcp`, `rsqrt`, `fneg` |
| Память | `ld`/`st` global и shared: `f32` `u32` `u64` `u8`/`s8` `u16`/`s16` `f16` `bf16` `v2.f32` |
| Shared memory | объявления `.shared`, `cvta.to.shared`, индексация `[reg+offset]` |
| Синхронизация | `bar.sync` → барьер рантайма |
| Warp | `shfl.{idx,bfly,up,down}` (в т.ч. с условной записью), `vote.*`, `activemask` |
| Атомики | `atom.add` `cas` `exch` `min` `max`, `red.add.*` → `atomicrmw` / `cmpxchg` |
| Матрицы | `mma.sync.aligned` (.row.col m16n8k16 f16/bf16/e4m3/e5m2/u8/s8 → f32/f16/s32, m16n8k8 f16/bf16 → f32/f16), `ldmatrix`/`stmatrix.sync.aligned.m8n8.x{1,2,4}[.trans].b16` |

`wmma.*`, `movmatrix`, `mma.sp` (sparse) и режимы округления `cvt.sat.*` пока
не реализованы и сообщаются как ошибки — громкий отказ, а не приближение.

### Документация

[docs/06-compiler-architecture.md](docs/06-compiler-architecture.md) — дескриптор
запуска, раскладка shared-памяти, модель `bar.sync`, атомики, снижение FP16/BF16
и планируемый путь SIMT → RVV.

### Проекты поверх ROUGE-V

Если вы поддерживаете проект, который использует ROUGE-V — форк, биндинг, бэкенд,
— укажите в своём README, что он создан не командой ROUGE-V и не связан с ней.

### Правовая рамка

ROUGE-V не содержит исходного кода, заголовков и бинарных файлов NVIDIA и не
занимается их реверс-инжинирингом. PTX, который он переводит, производится
собственным `nvcc` пользователя, по лицензии NVIDIA пользователя, на основе
опубликованной спецификации. Заголовочные файлы CUDA API в
`software/rouge-cuda/include/` написаны с нуля по публичной документации.
Правило clean-room закреплено в [CONTRIBUTING.md](CONTRIBUTING.md).

ROUGE-V не связан с NVIDIA, не одобрен и не поддерживается ею. CUDA и прочие
торговые марки принадлежат их владельцам и используются здесь только для
описания совместимости.

### Участие в проекте

Сначала прочитайте [CONTRIBUTING.md](CONTRIBUTING.md) — там описаны сборка,
работа с тестами и требование clean-room, которое распространяется на все
изменения. Issues и pull requests открыты.

### Благодарности

- **LLVM / MLIR** — IR, бэкенд и инфраструктура проходов, на которых это написано
- **RISC-V International** — спецификации ISA `rv64gcv` и Vector Extension
- **NVIDIA** — опубликованные спецификации PTX ISA и документация CUDA API, которые задают формат входа

### Лицензия

Apache 2.0 WITH LLVM-exception — та же лицензия, что у LLVM, чтобы код свободно
жил внутри LLVM и MLIR.
