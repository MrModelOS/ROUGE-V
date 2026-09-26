# Contributing to ROUGE-V

Спасибо за интерес к проекту. Ниже — минимальный набор правил для контрибьюторов.

## Сборка

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Требования: `clang` ≥ 17, `cmake` ≥ 3.20, `ninja`, `python3` (опционально для скриптов).

Отдельные компоненты:

```bash
# только компилятор (PTX -> LLVM IR -> нативный код)
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -S software/rouge-compiler
cmake --build build
ctest --test-dir build --output-on-failure

# CUDA-совместимый слой (CPU-эмуляция)
cmake -B build -G Ninja -S software/rouge-cuda
cmake --build build
ctest --test-dir build --output-on-failure
```

## Тесты

```bash
ctest --test-dir build --output-on-failure          # все тесты
ctest --test-dir build -R compiler_ptx_to_llvm -V   # фильтр по имени
./build/software/rouge-compiler/tests/test_ptx_to_llvm  # прямой запуск бинаря
```

CI гоняет `ctest --output-on-failure` на каждый push/PR (см. `.github/workflows/ci.yml`).

## Стиль кода

- C++17, `-Wall -Wextra`, без ворнингов.
- Форматирование — `clang-format` (LLVM style, 100 колонок). Перед коммитом:

  ```bash
  clang-format -i software/rouge-compiler/src/*.cpp software/rouge-compiler/include/**/*.h
  clang-format -i software/rouge-cuda/src/*.cpp software/rouge-ptx/src/*.cpp
  ```

- Имена: `snake_case` для функций/переменных, `CamelCase` для типов.
- Комментарии — по делу, без воды.

## DCO (Developer Certificate of Origin)

Каждый коммит должен содержать `Signed-off-by`:

```
Signed-off-by: Your Name <you@example.com>
```

Делается автоматически через `git commit -s`. Этим вы подтверждаете, что имеете право
внести вклад под лицензией проекта (Apache-2.0) и что вклад — ваша оригинальная работа
(см. https://developercertificate.org/).

## Clean-room правило

- **Запрещено** копировать код, заголовки или микроархитектурные детали из закрытых SDK NVIDIA (CUDA Toolkit, cuBLAS, cuDNN, драйвер) и любых других проприетарных источников.
- Допустимые источники: открытая документация PTX ISA (https://docs.nvidia.com/cuda/parallel-thread-execution/), открытые стандарты (LLVM, MLIR, RISC-V), собственные измерения поведения через чёрный ящик.
- Если сомневаетесь — откройте issue и обсудите до написания кода.
- Нарушения clean-room — блокер для мержа.

## Процесс

1. Форкните репозиторий, создайте ветку `feat/...` или `fix/...`.
2. Добавьте/обновите тесты.
3. Убедитесь что `cmake --build build && ctest` зелёный и `clang-format` применён.
4. Откройте PR с описанием мотивации и, если меняется архитектура, с обновлением `docs/06-compiler-architecture.md`.
5. Дождитесь ревью и CI.

## ADR

Любое решение, меняющее архитектуру компилятора, описывается в `docs/06-compiler-architecture.md`.
