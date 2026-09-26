# ROUGE-V — Quickstart

Тридцать секунд от клона до работающего перевода.

## Предпосылки

```bash
clang --version     # >= 17
cmake --version     # >= 3.20
ninja --version
```

`nvcc` не обязателен. Если он есть, включается тест с настоящим PTX от вендора.

## Сборка и тесты

```bash
git clone https://github.com/MrModelOS/ROUGE-V.git
cd ROUGE-V

cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build

ctest --test-dir build --output-on-failure                    # 16/16
ctest --test-dir software/rouge-cuda/build --output-on-failure  #  8/8
```

Оба набора должны пройти полностью: без них собранный код не запускается корректно.

## Как этим пользоваться

### 1. Проверить, переведётся ли ядро

Главный вопрос — «а моё ядро вы возьмёте?». Ответ даёт одна команда:

```bash
./build/rouge-compiler/rouge-run kernel.ptx
```

```
input: kernel.ptx
kernels found: 1
  @gemm_tile  params=3  instructions=67
  shared memory: 1024 bytes in 2 variable(s)
    shA          @   0     512 bytes  (.b16 x 256)
    shB          @ 512     512 bytes  (.b16 x 256)
translation: ok (10869 bytes of LLVM IR)

OK
```

Если инструкция вне поддерживаемого подмножества — будет **отказ, а не мусор**:

```
$ ./build/rouge-compiler/rouge-run tensor_core.ptx
rouge-run: REFUSED: unsupported PTX op 'mma.sync.aligned.m16n8k16...' @ 4
This instruction is outside the supported subset. Nothing was emitted;
nothing was approximated.
```

Это осознанное поведение. Неверный ответ хуже отказа.

### 2. Получить объектный файл

```bash
./build/rouge-compiler/rouge-run kernel.ptx -o kernel.o --print-ir
```

`--print-ir` дополнительно сохраняет переведённый LLVM IR. Объектный файл
собирается настоящим бэкендом, поэтому битая трансляция обнаружится здесь же.

### 3. Посмотреть IR глазами

```bash
./build/rouge-compiler/ptx2ir kernel.ptx kernel.ll
cat kernel.ll
```

### 4. Собрать под RISC-V

```bash
./build/rouge-compiler/ptx2ir kernel.ptx /tmp/kernel.ll
clang --target=riscv64-unknown-elf -march=rv64gcv -c /tmp/kernel.ll -o /tmp/kernel-rv64.o
```

### 5. Собрать под любую из четырёх платформ

Цель выбирается флагом `--target` у `ptx2ir` — это просто префикс аргументов,
остальная команда не меняется. Один и тот же PTX собирается под четыре платформы:

```bash
# 1) хост x86-64 — цель по умолчанию
./build/rouge-compiler/ptx2ir kernel.ptx /tmp/kernel.ll
clang -O2 -c /tmp/kernel.ll -o /tmp/kernel.o

# 2) RISC-V + векторное расширение
./build/rouge-compiler/ptx2ir --target riscv64-unknown-elf kernel.ptx /tmp/kernel-rv.ll
clang --target=riscv64-unknown-elf -march=rv64gcv -c /tmp/kernel-rv.ll -o /tmp/kernel-rv64.o

# 3) AMD RDNA 3
./build/rouge-compiler/ptx2ir --target amdgcn-amd-amdhsa kernel.ptx /tmp/kernel-amd.ll
clang --target=amdgcn-amd-amdhsa -mcpu=gfx1100 -c /tmp/kernel-amd.ll -o /tmp/kernel-amd.o

# 4) NVIDIA
./build/rouge-compiler/ptx2ir --target nvptx64-nvidia-cuda kernel.ptx /tmp/kernel-sm75.ll
clang --target=nvptx64-nvidia-cuda -march=sm_75 -c /tmp/kernel-sm75.ll -o /tmp/kernel-sm75.o
```

| `--target` | Платформа | Флаги бэкенда | Что на самом деле проверено |
|---|---|---|---|
| *(по умолчанию)* `x86_64-pc-linux-gnu` | хост x86-64 | — | собирается **и исполняется** — тесты `aot_native_*` |
| `riscv64-unknown-elf` | RISC-V + Vector | `-march=rv64gcv` | собирается в нативный объект |
| `amdgcn-amd-amdhsa` | AMD RDNA 3 | `-mcpu=gfx1100` | собирается в нативный объект, **не исполнялось** |
| `nvptx64-nvidia-cuda` | NVIDIA | `-march=sm_75` | собирается в нативный объект, **не исполнялось** |

Собираются все пять канонических ядер из `software/rouge-compiler/tests/kernels/`:
`vadd`, `block_reduce`, `atomic_reduce`, `fp16_reduce`, `gemm_tile`.

Об ограничении стоит сказать прямо: строки AMD и NVIDIA — утверждение о **сборке**,
а не о работе. Видеокарты в проекте нет, исполнение на GPU не проверялось, и никаких
цифр производительности из этих строк не следует. Настоящая проверка исполнения —
только на x86-64 через `ctest`; RISC-V, AMD и NVIDIA проверены как объекты, которые
принимает настоящий бэкенд LLVM.

Почему `--target` меняет содержимое IR, а не только строку `target triple`:
у AMD и NVPTX свои адресные пространства, и без их указания бэкенд не может
отличить global от shared — подробности в
[docs/06-compiler-architecture.md](docs/06-compiler-architecture.md).

## Откуда берётся PTX

Три обычных способа:

```bash
# 1. Собрать PTX из исходника CUDA
nvcc -arch=sm_75 -ptx kernel.cu -o kernel.ptx

# 2. Вытащить из fatbin, который лежит в уже собранной программе
cuobjdump -ptx ./my_app > kernel.ptx

# 3. Взять из сохранённой сборки в виде .ptx (часто лежит рядом с .so/.exe)
```

## Что дальше

Чтобы **запустить** переведённое ядро, нужен хост-драйвер: он заполняет дескриптор
запуска (`tid`/`ctaid`/`ntid`/`nctaid`, база shared-памяти, хендл барьера) и
запускает функцию. Рабочий пример — `software/rouge-compiler/tests/test_aot_vadd.cpp`:
минимальный драйвер на 40 строк. Устройство дескриптора описано в
[docs/06-compiler-architecture.md](docs/06-compiler-architecture.md).

Альтернативный путь, не требующий драйвера: `rouge-cuda` принимает PTX-ядро
через знакомый CUDA Driver API и исполняет его на хосте. Интерфейс —
`software/rouge-cuda/include/rouge/cuda.h`.

## Документы

| | |
|---|---|
| `README.md` | обзор и навигация |
| `docs/06-compiler-architecture.md` | архитектура компилятора |
| `CONTRIBUTING.md` | сборка, тесты, правило clean-room |
| `software/rouge-compiler/README.md` | транслятор и рантайм |
| `software/rouge-cuda/README.md` | слой совместимости |
