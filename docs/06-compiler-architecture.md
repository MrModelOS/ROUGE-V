# 06. Архитектура AOT-компилятора и транслятора

> **Статус:** действует (рабочая веха: ptx2ir + нативный AOT-запуск;
> shared memory, `bar.sync`, атомики и FP16/BF16 в обоих путях исполнения;
> MLIR-контур GPU→Vector стартовал).
> **Код:** `software/rouge-compiler`, общий фронтенд `software/rouge-ptx`.

## Задача

Перехватить код CUDA на этапе промежуточного представления (IR) и
скомпилировать его под векторизованное ядро RISC-V **без эмуляции
runtime-инструкций**. Главный перф-путь — нативный бинарник, а не
интерпретатор.

## Конвейер компилятора: 4 уровня трансформации

```
[ CUDA Source / NVVM IR / PTX ]
              │
              ▼
    1. CUDA/PTX Parser (LLVM Front-End)
              │
              ▼
    2. MLIR Dialect Conversion (GPU/NVVM -> Vector/Linalg -> ROUGE Dialect)
              │
              ▼
    3. Hardware-Aware Optimization Pass (Memory Wall & Layouts)
              │
              ▼
    4. RISC-V LLVM Backend (+ Custom Vector/Matrix ISA Extensions)
              │
              ▼
[ Native ROUGE-V Binary + Runtime API ]
```

### Шаг 1. Перехват и парсинг (Front-End)

Писать парсер CUDA C++ с нуля не нужно: NVIDIA сама компилирует `.cu` через
nvcc в **NVVM IR** (высокоуровневый LLVM IR) или текстовый **PTX**.

- **NVVM IR Reader:** фронтенд принимает бинарный LLVM IR, генерируемый
  библиотеками (PyTorch через torch.compile / Triton) или nvcc — это штатный
  вход LLVM, читается без собственного парсера.
- **PTX JIT Parser:** для бинарников с готовым PTX работает наш транслятор
  PTX → LLVM IR. PTX — виртуальный ассемблер с регистровым файлом, который
  легко мапится на типизированный LLVM SSA-граф.
- **Реализация (работает):** `ptx2ir` в `software/rouge-compiler`
  транслирует PTX-подмножество (арифметика u32/s64/f32 и f16/bf16, cvt, setp,
  ld/st.global, ld/st.shared, cvta.to.shared, bar.sync, атомики
  `atom.add`/`red.add`, ветвления и предикаты) в текстовый LLVM IR. Кернел
  становится `define void @<kernel>(... i32 %argN, ptr %launch)`, где
  `%launch` — дескриптор запуска: 12×i32 спецрегистров (tid/ctid/ntid/nctaid ×
  xyz, смещения 0..44), `i64`-указатель на scratchpad блока (.shared, смещение
  48) и `i64`-хендл барьера блока (смещение 56).

### Shared memory и барьеры (реализованный прототип)

Поддержка `.shared` и `bar.sync` — фундамент, без которого не работает ни
блочная редукция, ни тайлинг матричного умножения.

- **Модель памяти (Scratchpad):** декларации `.shared .align N .b* var[K];`
  раскладываются парсером в единый по-блочный scratchpad-буфер (выравнивание
  имеет padding; итоговый размер — `PtxFunction::sharedSize`). Адрес
  .shared-переменной — **база scratchpad + смещение** (identity-маппинг), в
  обоих путях исполнения:
  - **AOT (ptx2ir):** база читается из launch-дескриптора (смещение 48);
    драйвер выделяет буфер на блок. `cvta.to.shared` ⇒ `add i64` к базе;
    `ld/st.shared` (форма `[%rd]` или `[sym±const]`) ⇒ load/store по
    вычисленному адресу — тот же конвейер, что у `ld/st.global`, но через
    scratchpad.
  - **Интерпретатор:** по-блочный `std::vector<uint8_t>` scratchpad, адреса —
    host-указатели (identity), синхронизацию исполняет CTA-секвенсер (ниже).
- **Барьер `bar.sync`:** в AOT-пути — вызов `__rouge_syncthreads(ptr %launch)`
  из C-API runtime (`rouge-compiler/runtime/rouge_runtime.cpp`): на хост-стенде
  это `pthread_barrier_wait`, на целевом RISC-V — кастомный межядерный
  барьер/`fence` (Шаг 2).
- **SIMT-семантика в интерпретаторе:** потоки блока исполняются **в ногу**
  (lockstep CTA-секвенсер в `execute_kernel`): все живые потоки блока шагают
  по одной инструкции за раунд; когда каждый живой поток упёрся в барьер —
  барьер отпускается. Это честно моделирует видимость записи в shared-память
  «до» `bar.sync` для чтений «после». (Наивное последовательное исполнение
  потоков неверно: поток 0 завершал кернел раньше, чем поток 1 записывал
  данные.)
- **Проверка (один кернел — два пути):** `block_reduce.ptx` — блочная
  редукция через shared-память + `bar.sync`. Один и тот же PTX-файл
  исполняется в AOT (`aot_native_block_reduce`: 16 блоков × 256 потоков,
  pthread-барьер) и в интерпретаторе (`driver_block_reduce`, CTA-секвенсер);
  оба дают одинаковые суммы блоков.

### Атомики (`atom.*` / `red.*`) — реализовано

Следующий блок к реальным GEMM: глобальное накопление результата и
claim-проверки без гонок.

- **Набор:** `atom.add.{u32,s32,u64,s64,f32}` (возвращает старое значение) и
  `red.add.*` (без возврата). Квалификаторы scope/space (`.global`, `.cta`) и
  семантика `.relaxed` принимаются; память в обоих путях — единая.
  Всё остальное (`atom.exch`, `atom.cas`, `atom.max/min`, compare-and-swap) —
  **явная ошибка транслятора**.
- **AOT:** `atomicrmw add` / `atomicrmw fadd` с порядком `monotonic`
  (эквивалент PTX `relaxed`). На x86-64 это `lock xadd`/CAS-цикл, на RISC-V —
  AMO; то есть настоящие атомарные инструкции, без эмуляции. Для `red.*`
  результат отбрасывается (операция всё равно атомарна).
- **Интерпретатор:** атомарность тривиальна (потоки блока исполняются
  последовательно), тип- и wrap-семантика повторяет AOT.
- **Проверка (один кернел — два пути):** `atomic_reduce.ptx` — все потоки
  атомарно копят в общий f32-сумматор и u32-счётчик, а `atom.add.u32` на
  приватном слоте проверяет возвращаемое старое значение (claim-and-clear).
  AOT (`aot_native_atomic_reduce`: 16×256 настоящих потоков, гонка на одних
  кэш-линиях) и интерпретатор (`driver_atomic_reduce`) дают одинаковый результат.

### FP16 / BF16 — реализовано

- **Операции:** `add/sub/mul/div.f16`, `neg.f16`, `fma.rn.f16`,
  `add/sub/mul.rn.bf16`, `fma.rn.bf16`; конверсии `cvt.f32.f16`,
  `cvt.rn.f16.f32`, `cvt.rn.bf16.f32`, `cvt.f32.bf16`, `cvt.{u32,s32}.{f16,bf16}`
  и обратные; `ld/st` типов `.f16/.b16/.bf16` (16-битные регистры PTX `.b16` →
  `i16` allocas в IR).
- **Семантика:** вычисление в f32 с **единственным** сужением до формата
  (RNE, `fptrunc` в IR) — это побитово совпадает с нативной арифметикой
  binary16/bfloat16 для add/sub/mul/div. На RISC-V при наличии Zfh/Zfbf
  целевые инструкции заменят эту схему (TODO в бэкенде), а на хосте обе
  схемы совпадают — поэтому интерпретатор и AOT дают **бит-идентичные**
  16-битные паттерны (проверено тестом `fp16_reduce` в обоих путях).
- **Проверка:** `fp16_reduce.ptx` — f16-тайл в shared + `bar.sync`,
  f16/bf16 round-trip конверсий, `add.f16` в петле, `fma.rn.f16`,
  `add.rn.bf16` на выходе.

### Экспорт раскладки `.shared` в runtime — реализовано

Драйвер больше не знает размеров scratchpad: кернел **сам описывает** свою
.layout-память.

- **Что эмитится:** для каждого транслированного кернела ptx2ir выпускает
  таблицу `.shared`-переменных (`@__rouge_<kernel>_shared_vars`: имя, смещение,
  размер, выравнивание, размер элемента, количество) и функцию
  `i32 @__rouge_<kernel>_query(ptr %info)`, заполняющую `RougeKernelInfo`
  (`rouge-runtime.h`): общий размер scratchpad + указатель на таблицу.
- **Как это используется:** `rouge_kernel_info_query()` → `rouge_smem_alloc()`
  (нулевой буфер на блок) / `rouge_smem_offset(info, "smem")`. Тест-драйвер
  `test_aot_block_reduce.cpp` больше не содержит ручного `kScratch=1024`:
  размер приходит из PTX. Смещения ABI зафиксированы `static_assert`-ами в
  рантайме.
- **Почему это важно для продукта:** C-API-рантайм сможет сам выделять
  scratchpad и проверять символы при загрузке модуля — то, что потребуется
  rouge-cuda для запуска AOT-бинарников из PTX без знания о ядре.

### Шаг 2. MLIR-трансформация (Middle-End)

Ключевой этап — отображение иерархии CUDA (Grid → Block → Thread) на
архитектуру ROUGE-V.

- **Развёртывание потоков (Thread Block Lowering):** поток CUDA исполняется
  в модели SIMT (Single Instruction, Multiple Threads). На ROUGE-V группа из
  32/64 CUDA-потоков (Warp) объединяется в один векторный lane шириной
  `VLEN` через векторные регистры RISC-V Vector Extension (RVV).
- **Преобразование памяти:**
  - `__shared__` CUDA-память → физический Scratchpad SPM / L1-кэш ядра RISC-V;
  - `__global__` → DMA-трансферы между HBM/DDR и локальной памятью ядра.
- **Синхронизация:** `__syncthreads()` и warp-барьеры транслируются в
  аппаратные барьеры RISC-V (`fence` и кастомные сигналы межядерной
  синхронизации).

> **Статус:** диалекты MLIR — контур стартовал. `software/rouge-compiler/mlir`
> (`rouge-opt --rouge-simt-access-report`) измеряет SIMT-паттерны
> (thread/block ids, барьеры, global/shared-трафик, атомики, vector-операции)
> — вход для SIMT→RVV-векторизатора. Скалярный ptx2ir даёт корректный код
> (alloca-стиль, уже оптимизируется LLVM mem2reg); MLIR-слой заменит его
> векторизацией. Диаграмма выше — целевая архитектура, реализация движется
> по шагам (`mlir/README.md`).

### Шаг 3. Оптимизационный пайплайн (Hardware Passes)

Чтобы не терять производительность на обращении к памяти, пишутся кастомные
проходы (Passes) в MLIR:

- **Tiling & Loop Unrolling:** автоматическая разбивка глобальных матричных
  операций под размеры локальных SRAM-буферов (Scratchpad).
- **Memory Coalescing & Layouts:** преобразование обращений из формата CUDA
  в оптимизированный формат для векторных загрузок `vle32.v` / `vse32.v`
  без конфликтов банков памяти.
- **Double Buffering (DMA Ping-Pong):** сокрытие задержек глобальной памяти:
  пока вычислительные блоки считают блок N, DMA-контроллер параллельно
  качает блок N+1.

### Шаг 4. Генерация кода (Back-End) и Runtime

- **LLVM Backend для RISC-V:** добавляем в LLVM поддержку нашего ассемблерного
  расширения (`rv64gcv_xrouge` — базовый RV64 + Vector + кастомные инструкции
  матричного умножения `matmul.mma`). Тот же IR, который генерирует ptx2ir,
  уже собирается штатным бэкендом `riscv64-unknown-elf -march=rv64gcv`
  (проверяется тестом `compiler_rvv_backend`).
- **C-API Runtime (замена libcuda.so / libcudart.so):** открытая обёртка-
  заглушка над вызовами CUDA C-API:
  - `cudaMalloc()` → выделение в адресном пространстве ROUGE-V;
  - `cudaMemcpy()` → запуск DMA-контроллера платы;
  - `cudaLaunchKernel()` → загрузка бинарника в Command Queue и сигнал старта
    ядра.

## Текущее состояние (рабочая веха)

| Компонент | Статус | Как проверить |
|---|---|---|
| Общий PTX-фронтенд `rouge-ptx` (парсер + типы регистров, f16/bf16) | ✅ | сборка обоих потребителей |
| `ptx2ir` — PTX → LLVM IR (+ атомики, f16/bf16, экспорт `.shared`-метаданных) | ✅ | `ctest compiler_ptx_to_llvm_atomics_fp16` |
| Нативный AOT-запуск (clang → объект → драйвер → проверка) | ✅ | `ctest aot_native_vadd` |
| Shared memory + `bar.sync` в AOT (блок-редукция, 16×256) | ✅ | `ctest aot_native_block_reduce` |
| Shared memory + `bar.sync` в интерпретаторе (CTA-секвенсер) | ✅ | `ctest driver_block_reduce` (rouge-cuda) |
| Атомики `atom.add`/`red.add` (AOT atomicrmw + интерпретатор) | ✅ | `ctest aot_native_atomic_reduce` / `driver_atomic_reduce` |
| FP16/BF16 (half/bfloat, fma, cvt, ld/st, shared-тайл) | ✅ | `ctest aot_native_fp16_reduce` / `driver_fp16_reduce` |
| Экспорт `.shared`-раскладки в runtime (`__rouge_*_query`) | ✅ | `test_aot_block_reduce` без `kScratch` + `rouge_runtime.h` |
| RVV-кросс-проверка (`rv64gcv`) — 3 ядра | ✅ | `ctest compiler_rvv_backend_*` |
| MLIR-контур: `rouge-simt-access-report` (GPU→Vector) | ✅ контур | `cmake -DROUGE_ENABLE_MLIR=ON` + `rouge-opt --rouge-simt-access-report` |
| MLIR Dialect Conversion GPU→Vector/Linalg→ROUGE (векторизация) | 📋 план | — |
| Hardware-Aware Passes (tiling/coalescing/double-buffer) | 📋 план | — |
| Backend-расширения `rv64gcv_xrouge` (`matmul.mma`) | 📋 план | — |
| Runtime на базе нативных кернелов (Command Queue, DMA) | 📋 план | — |

## Неподдерживаемое подмножество PTX (явная ошибка, не тихий мискомпил)

Warp-shuffle (`shfl.*`), тензорные инструкции, `atom.exch`/`atom.cas`/
`atom.min/max`, 16-битные векторные формы (`.f16x2`/`.bf16x2`) — транслятор
возвращает ошибку с указанием инструкции (честный fail, без тихого
мискомпила). Атомики `atom.add`/`red.add` и FP16/BF16 уже поддержаны
(см. выше). Порядок дальнейшего расширения — в
`software/rouge-compiler/README.md` и дорожной карте (задачи Phase 0).

## Связанные документы

- [03. Архитектура ПО](./03-software-architecture.md) — место компилятора в стеке
- [05. Дорожная карта](./05-roadmap.md) — задачи Phase 0–4
- [02. Архитектура железа](./02-hardware-architecture.md) — векторное ядро, SPM, DMA