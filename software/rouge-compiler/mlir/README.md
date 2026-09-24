# MLIR-контур ROUGE-V (стадия 2 конвейера: GPU → Vector → RVV)

Стадия 1 (`ptx2ir`: PTX → LLVM IR → нативный код) живёт в родительском каталоге
и собирается везде. Этот каталог — **стадия 2**: lowering SIMT-семантики в
векторные операции RVV через MLIR.

## Статус: контур (конечная точка каркаса)

Что уже есть:

- `lib/Transforms/SimtAccessReport.cpp` — проход `--rouge-simt-access-report`:
  **анализ** SIMT-паттернов (thread/block id, барьеры, global/shared traffic,
  атомики, vector-операции) по функциям. Ничего не переписывает: его задача —
  сделать измеримым то, под что проектируется векторизатор (unit-stride
  участки между барьерами).
- `tools/rouge-opt/rouge-opt.cpp` — драйвер вроде `mlir-opt` с нашими проходами.
- `test/simt_access_report.mlir` — тестовый вход (GEMM-тайл + чистый copy).

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
