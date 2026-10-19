# MLIR-контур ROUGE-V (стадия 2 конвейера: GPU → Vector → RVV)

Стадия 1 (`ptx2ir`: PTX → LLVM IR → нативный код) живёт в родительском каталоге
и собирается везде. Этот каталог — **стадия 2**: lowering SIMT-семантики в
векторные операции RVV через MLIR.

## Статус: контур собирается и проверяется

Что уже есть:

- `lib/Transforms/SimtAccessReport.cpp` — проход `--rouge-simt-access-report`:
  **анализ** SIMT-паттернов (thread/block id, барьеры, global/shared traffic,
  атомики, vector-операции) по функциям. Ничего не переписывает: его задача —
  сделать измеримым то, под что проектируется векторизатор (unit-stride
  участки между барьерами).
- `tools/rouge-opt/rouge-opt.cpp` — драйвер вроде `mlir-opt` с нашими проходами.
- `test/simt_access_report.mlir` — тестовый вход (GEMM-тайл + чистый copy);
  `test/simt_access_report_atomic.mlir` — атомики (синтаксис `atomic_rmw`
  изменился в MLIR 23, поэтому ячейка включается только на MLIR >= 23).

Проверено на MLIR 23.1.3 (официальный релиз LLVM, ровно та версия, что стоит
в системе): контур собирается, `rouge-opt` печатает отчёт по функциям, обе
ячейки ctest зелёные. Пример вывода:

```
rouge-simt-access-report: module '(anonymous)'
  scale_copy: loads=1 stores=1 vector=0 atomics=0 barriers=0 ids=0
  gemm_tile: loads=2 stores=2 vector=2 atomics=1 barriers=1 ids=2
rouge-simt-access-report: lowering plan hints
  scale_copy: unit-stride candidate — SIMT->RVV vectorizable
  gemm_tile: mixed access — vectorize only the runs between barriers
```

Чего ещё нет (следующие итерации, по порядку):

1. `gpu-launch-to-scf`: `gpu.launch_func` + `gpu.thread_id` → `scf` с
   явными измерениями (блок = итерация, поток = lane).
2. `rouge-unit-stride-to-vector`: серии `memref.load/store` с шагом 1 →
   `vector.load/store` (это и есть то, что измеряет проход выше).
3. `vector-to-rvv`: `vector.*` поверх `memref` → расширения RVV
   (`riscv.vle`, `riscv.vse`, `riscv.vfmacc`) через таргет-специфичный lowering.
4. Общий режим: один PTX/MLIR фронтенд, ноль эмуляции инструкций.

## Сборка

Нужен LLVM с MLIR (в дереве LLVM или готовый пакет):

```sh
cmake -S . -B build -DROUGE_ENABLE_MLIR=ON -DMLIR_DIR=/path/to/llvm/lib/cmake/mlir
cmake --build build
ctest --test-dir build
```

Проверенная конфигурация (без root, только распакованный релиз LLVM):

```sh
curl -L -o llvm.tar.zst https://github.com/llvm/llvm-project/releases/download/llvmorg-23.1.3/LLVM-23.1.3-Linux-X64.tar.zst
zstd -dc --long=30 llvm.tar.zst | bsdtar -xf - -C ~/.local/llvm-mlir --strip-components 1
cmake -S software/rouge-compiler -B build-mlir -DROUGE_ENABLE_MLIR=ON \
      -DMLIR_DIR=$HOME/.local/llvm-mlir/lib/cmake/mlir
cmake --build build-mlir
ctest --test-dir build-mlir -R mlir
```

Две детали, на которые стоит смотреть: `--long=30` у `zstd` (фрейм требует
окно в 1 ГБ) и подстановка `zstd::libzstd_static` на shared-вариант, если
статической zstd в системе нет (LLVM в своих CMake-экспортах на неё
ссылается).

Без MLIR проходы не собираются: таргет выключен по умолчанию, `ctest`
просто не содержит MLIR-тестов. Это осознанно — стадия 1 (ptx2ir) не должна
требовать MLIR, чтобы проверяться на любой машине с clang.

## Почему так

Сырой PTX — это SIMT: один поток = один элемент, ветвления по `tid`.
RVV — это data-parallel: один поток = вектор. Прямой транслятор будет
«наивным и медленным» (одна инструкция на элемент). Правильный путь —
**сначала измерить** форму обращений к памяти (проход выше), потом сливать
соседние обращения в векторные операции, и только потом отображать их в RVV.
Отсюда порядок: анализ → GPU→SCF → unit-stride→vector → vector→RVV.
