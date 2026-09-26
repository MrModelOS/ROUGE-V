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

```sh
./build/rouge-compiler/ptx2ir \
    software/rouge-compiler/tests/kernels/vadd.ptx /tmp/vadd.ll

clang -O2 -c /tmp/vadd.ll -o /tmp/vadd.o                  # хост
clang --target=riscv64-unknown-elf -march=rv64gcv \
      -c /tmp/vadd.ll -o /tmp/vadd-rv64.o                 # RISC-V + Vector
```

```
$ ctest --test-dir build
...
 5/16 aot_native_block_reduce ..............   Passed   smem 1024B 16x256
 6/16 aot_native_atomic_reduce .............   Passed   sum=4608 count=4096
 7/16 aot_native_fp16_reduce ...............   Passed   32x32 f16/bf16
 8/16 aot_native_gemm_tile .................   Passed   2x2 blocks 16x16
13/16 compiler_nvcc_ptx ....................   Passed   nvcc -> ptx2ir -> clang
100% tests passed out of 16
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

Оба пути исполнения построены на одном парсере. Это сделано намеренно: одна
грамматика, один набор правил для регистров, параметров, раскладки `.shared` и
меток — чтобы пути не расходились. Тесты строятся вокруг требования, что на
каждом ядре они дают **бит-в-бит одинаковый** результат.

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

Тензорные инструкции (`mma`), векторные формы `f16x2` и режимы округления
`cvt.sat` пока не реализованы и сообщаются как ошибки.

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
