# rouge-numerics — своя открытая математика вместо cuBLAS

**Что это.** Своя открытая математика ROUGE-V (Волна 2 роадмапа): блочный
FP32 CPU-GEMM с автотюнером, GEMM на половинных/FP8 входах, свёртка
(im2col+GEMM, Winograd F(2x2,3x3), автотюнер метода), блочный attention с
онлайн-softmax, свой radix-2 FFT, CSR SpMV, детерминированный RNG и шимы
cuBLAS/cuDNN поверх своего кода. Реализация целиком своя: ни cuBLAS, ни BLAS,
ни Eigen, ни FFTW внутри — только стандартная библиотека C++17 и
target-атрибуты компилятора. Задача — снять зависимость от проприетарной
математики в собственном стеке.

**Чего это НЕ делает.**

- Это **не замена cuBLAS по скорости**. Это CPU-ядра. Они считают на CPU
  (в том числе на эмуляторе) и не имеют ничего общего с тензорными ядрами GPU.
  Любая цифра в этом README получена замером на конкретной машине и относится
  только к ней.
- Это **не трансцендентная функция**: GEMM, свёртка, attention, FFT, sparse,
  RNG — и всё.
- Это **не fp64**: только float32 накопление (входы — f32/f16/bf16/e4m3).
- Шимы покрывают **только f32-подмножество** (`cublasSgemm`, conv-forward
  NCHW); остальное отсутствует намеренно — громкая ошибка линковки вместо
  тихого вранья.
- FFT — только степени двойки, иначе громкий `-1`.

## Сборка и тесты

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -S software/rouge-numerics
cmake --build build
ctest --test-dir build --output-on-failure
```

Сборка самодостаточная: библиотека не подключена к корневому `CMakeLists.txt`
ROUGE-V, в дереве лежит свой проект.

Что проверяют тесты:

| Тест | Что делает |
| --- | --- |
| `gemm` | Каждый конфиг-кандидат на восьми размерах против наивного эталона (naive triple loop), плюс автотюнер; печатает GFLOP/s по размерам |
| `gemm_isa_base` / `gemm_isa_avx2` / `gemm_isa_avx512` | Те же конфиги, но для **каждого** собранного варианта ISA — процессор умеет не всё, без этого проверялся бы ровно один путь |
| `gemm_fp8` | Квантование e4m3 (точечные векторы + roundtrip всех 256 значений), точный GEMM на целых 1..8 бит-в-бит, дробные данные 2% против шкалы |
| `gemm_f16` | Точечные значения f16/bf16 (1.0, −0, Inf, NaN), roundtrip целых 1..32, GEMM обоих форматов против f32-эталона (допуск 1e-2 — ошибка квантования входов) |
| `conv` | Свёртка NCHW: ручной случай 3x3 (−6) плюс 6 случайных конфигураций (stride/pad/bias, неквадратные ядра) против независимого прямого эталона |
| `winograd` | Winograd F(2x2,3x3): ручной случай (единичное ядро — выход равен входу) + 5 случайных конфигураций (чёт/нечет, bias) против im2col, допуск 1e-4; плюс автотюнер метода |
| `attention` | Блочный attention против наивного двухпроходного эталона в double: обычный, causal, causal+сдвиг, Skv=130 (пересчёт m/l через границу блока), допуск 1e-5 |
| `fft_sparse_rng` | FFT n=1..256 против наивного DFT + roundtrip + отказ на n=3; SpMV на случайной разреженности против плотного эталона бит-в-бит + проверки границ; RNG: детерминизм, uniform в [0,1), normal mean/var на 100k |
| `shims` | `cublasSgemm` все 4 комбинации N/T против эталона в double (column-major, alpha/beta) + отказы; `cudnnConvolutionForward` оба алгоритма против `rouge_conv2d_f32` + отказы (HALF, dilation) |
| `bench_blas` | Замер нашего GEMM против системного BLAS и наивной петли (корректность обязательна, скорость — отчёт); см. «Замеры» ниже |

Проверенные размеры, в том числе не кратные плитке: `1x1x1`, `3x5x7`, `17x33x65`,
`128x128x128`, `200x100x50`, `64x64x128`, `129x131x127`, `256x192x128`.

Погрешность: `max|C − ref| / max(‖ref‖∞, 1)`. Единица в знаменателе защищает от
деления на почти-нулевую сумму при `K=1` и не маскирует ошибку — элементы
порядка единицы. Допуск `1e-5`, измеренная худшая погрешность по всем конфигам
и размерам — порядка `3e-7` (то есть ~30x запас), что обычное накопление
ошибки округления float32 при суммировании.

## API

Всё в `include/rouge/numerics.h`, `extern "C"` — заголовок собирается и из C.

```c
typedef struct RougeGemmConfig { int tile_m, tile_n, tile_k, unroll, split_k; } RougeGemmConfig;
typedef struct RougeGemmProfile { RougeGemmConfig cfg; double gflops; long long iters; } RougeGemmProfile;

/* Обычный блочный GEMM: C[M][N] = A[M][K] * B[K][N] (row-major), с FMA. */
void rouge_gemm_f32(int M, int N, int K, const float* A, const float* B, float* C,
                    const RougeGemmConfig* cfg);   /* cfg == NULL -> по умолчанию */

RougeGemmConfig rouge_gemm_default_config(void);
int      rouge_gemm_candidate_count(void);
RougeGemmConfig rouge_gemm_candidate(int index);

/* Скорость одного конфига на этих данных, GFLOP/s. */
double rouge_gemm_flops_per_sec(int M, int N, int K, const float* A, const float* B, float* C,
                                const RougeGemmConfig* cfg);

/* Подобрать конфиг замером, применить, вернуть профиль (конфиг + GFLOP/s). */
RougeGemmProfile rouge_gemm_tuned(int M, int N, int K, const float* A, const float* B, float* C);

const char* rouge_gemm_isa_variant(void);  /* "base+sse2" | "avx2+fma" | "avx512+fma" */

/* FP8 GEMM: C[M][N] f32 = A[M][K] e4m3 * B[K][N] e4m3, накопление f32.
 * OCP OFP8 E4M3; decode в f32 + тот же блочный GEMM (автотюнер shared). */
void rouge_gemm_e4m3(int M, int N, int K, const uint8_t* A, const uint8_t* B,
                     float* C, const RougeGemmConfig* cfg);
uint8_t rouge_quantize_e4m3(float f);   /* RNE, satfinite в знаковый MAX */
float rouge_dequantize_e4m3(uint8_t u);

/* Свёртка 2D NCHW f32: im2col + свой блочный GEMM. bias может быть NULL. */
void rouge_conv2d_f32(int N, int Cin, int H, int W, int Cout, int KH, int KW,
                      int stride, int pad, const float* X, const float* Wt,
                      const float* bias, float* Y,
                      const RougeGemmConfig* cfg);

/* Winograd F(2x2,3x3) для 3x3 stride 1 pad 1. Успех -> 0, иначе -1. */
int rouge_conv_winograd3x3_f32(int N, int Cin, int H, int W, int Cout,
                               const float* X, const float* Wt,
                               const float* bias, float* Y);

typedef struct RougeConvProfile { int method; double millis; } RougeConvProfile;
/* Автотюнер свёртки: замер im2col против Winograd; в Y результат победителя.
 * method: 0=im2col, 1=winograd, -1=ошибка. */
RougeConvProfile rouge_conv_tuned(int N, int Cin, int H, int W, int Cout,
                                  int KH, int KW, int stride, int pad,
                                  const float* X, const float* Wt,
                                  const float* bias, float* Y);

/* Attention: O[S][D] = softmax(Q·K/scale)·V, блочный проход Bt=64.
 * causal маскирует t > s+kv_off; scale<=0 -> 1/sqrt(D). */
void rouge_attention_f32(int S, int Skv, int D, const float* Q, const float* K,
                         const float* V, float* O, float scale, int causal,
                         int kv_off);

typedef struct RougeComplex { float re, im; } RougeComplex;
/* FFT radix-2: inverse=0 forward, иначе inverse с /N. Успех -> 0. */
int rouge_fft_f32(int n, const RougeComplex* in, RougeComplex* out, int inverse);

/* CSR SpMV: y = A*x. Успех -> 0, нарушение границ -> -1. */
int rouge_csr_spmv_f32(int nrows, int ncols, const int* rowptr,
                       const int* colidx, const float* values, const float* x,
                       float* y);

typedef struct RougeRng { uint64_t state; int have_next; float next_gauss; } RougeRng;
void rouge_rng_seed(RougeRng* rng, uint64_t seed);
uint64_t rouge_rng_next_u64(RougeRng* rng);
float rouge_rng_uniform_f32(RougeRng* rng);   /* [0,1) */
float rouge_rng_normal_f32(RougeRng* rng);    /* N(0,1), Box-Muller */
```

Шимы чужих API — отдельные shared-библиотеки поверх этого кода
(`rouge-cublas`, `rouge-cudnn`; интерфейсы по открытой документации,
реализация своя — подход Wine):

```c
#include <rouge/cublas_shim.h>
cublasCreate_v2(&h);
cublasSgemm_v2(h, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, A, lda,
               B, ldb, &beta, C, ldc);   /* column-major, только f32 */

#include <rouge/cudnn_shim.h>
cudnnCreate(&h); /* + дескрипторы NCHW/FLOAT + cudnnConvolutionForward */
```

## Замеры (i5-1135G7, путь `avx512+fma`, минимум из 5 прогонов wall-clock)

`bench_blas`, кэш-резидентные размеры (это важно: матрицы влезают в кэш,
поэтому цифры — не пик железа и не заявление про большие панели):

| Размер | rouge_gemm_f32 | системный cblas_sgemm | наивная петля |
| --- | --- | --- | --- |
| 64×64×64 | ~85 GFLOP/s | ~3.5 GFLOP/s | ~2.3 GFLOP/s |
| 128×128×128 | ~88 GFLOP/s | ~4.3 GFLOP/s | ~2.2 GFLOP/s |
| 256×192×128 | ~93 GFLOP/s | ~4.7 GFLOP/s | ~2.3 GFLOP/s |

Честные оговорки: системный BLAS здесь — эталонный reference BLAS
(`libcblas`, не OpenBLAS; его никто не настраивал), поэтому разрыв в 20× —
это «мы против ненастроенного эталона», а не «мы против OpenBLAS».
Сравнение с OpenBLAS/oneDNN/FFTW не проводилось (зависимостей нет, цифры без
замера не пишем). Свёртка 2×16×32×32→32ch 3×3 s1p1: im2col 0.47 мс против
Winograd 0.66 мс — Winograd корректен (отн. погрешность 2e-7), но на этом CPU
проигрывает нашему же AVX512-GEMM: поточечный цикл скалярный, а микроядро
GEMM векторизовано. Автотюнер `rouge_conv_tuned` для того и нужен — он
выбирает победителя замером, а не верой.

Значения полей конфига <= 0 берутся из конфига по умолчанию, поэтому
`{0,0,0,0,0}` — корректный запрос «по умолчанию». Плитки ограничены сверху
(256), `unroll` допускает только 1/2/4, `split_k` — верхняя граница числа
частей K (для `K < split_k` реально частей `min(split_k, K)`).

Поля конфига:

- `tile_m` — высота панели A, `tile_n` — ширина панели B, `tile_k` — шаг по K;
- `unroll` — во сколько раз микроядро разворачивает цикл по K;
- `split_k` — на сколько частей делится K, каждая часть считается своим
  `std::thread`, результаты складываются.

## Устройство ядра

`src/gemm.cpp` + `src/gemm_kernel.inc` + `src/gemm_detail.h`:

1. **Упаковка панелей.** `A` упаковывается в `tile_m x tile_k` построчно, `B` —
   в `tile_k x tile_n` построчно. Микроядру удобно: `A[i][k]` — broadcast-скаляр,
   `B[k][j]` — вектор, обе стороны идут подряд.
2. **Двойная буферизация по K.** Следующая k-панель упаковывается в запасной
   буфер **до** вычисления текущей, поэтому данные, которые ещё читаются, не
   затираются.
3. **Микроядро `MR x NR`.** Полосы C держится в регистрах, цикл по K
   разворачивается в `unroll` шагов. Все индексы полного блока известны
   на этапе компиляции, поэтому это настоящий регистровый блок, а не
   «цикл, который компилятор, может быть, развернёт».
4. **Края** (размеры, не кратные плитке) считает отдельный edge-ядро: там
   границы известны только в рантайме, полосы в регистрах не удержать.
5. **`split_k`.** K делится на части, каждая считается своим `std::thread` в
   свой накопитель `[M][N]`, в конце части складываются в `C`.

**Варианты ISA.** Один и тот же `.inc` включается три раза — как `base+sse2`
(без целевых флагов, работает везде), `avx2+fma` и `avx512+fma` (через
`__attribute__((target(...)))`). Выбор — в рантайме по
`__builtin_cpu_supports`. Сравниваются замеры одного и того же алгоритма, а не
разные алгоритмы. Переменная окружения `ROUGE_NUMERICS_ISA=base|avx2|avx512`
принудительно выбирает вариант; если процессор не умеет запрошенный, просьба
отклоняется (риска исполнить неподдерживаемую инструкцию нет) — тесты на это
смотрят и печатают `SKIP`.

## Автотюнер

`src/autotune.cpp` перебирает 7 кандидатов (`tm/tn` 32…128, `tk` 32…64,
`unroll` 1…4, `split_k` 1/2/4) и выбирает самый быстрый по wall-clock. Правила,
которых он придерживается:

- замер идёт на настоящих данных `A`/`B`/`C` вызывающего, результат пишется в `C`;
- **кэша нет**: каждый прогон действительно считает;
- перед замером один прогон-разогрев, его время не учитывается;
- из нескольких прогонов берётся минимум — так шум планировщика не завышает
  оценку кандидата;
- в профиль попадает конфиг, который реально применялся, и
  `2*M*N*K / время`.

## Узкие места (честно)

- **Однопоточный расчёт по M/N.** Параллелизм есть только по K (`split_k`), и
  его даёт лишь несколько процентов на этой машине. Матрица 1024×1024×512
  целиком в одном потоке упирается в пропускную способность памяти, а не в
  вычисления.
- **Нет SIMD-интринсиков, только автовекторизация компилятора.** Микроядро
  написан обычным C++; качество векторизации целиком в руках GCC/Clang. Ручной
  AVX-512 с более широким регистровым блоком (8×64 вместо 4×32) почти наверняка
  дал бы больше — но это следующий шаг, а не то, что здесь написано.
- **Упаковка дублируется между тайлами.** Панель `B` для тайла по N
  переупаковывается для каждого тайла по M. На больших K это заметная доля
  работы; классическое решение — паковать B один раз на весь проход по M.
- **Крайние тайлы считаются заметно медленнее** полных: edge-ядро работает по
  одному `float` в регистре.
- **Набор кандидатов фиксирован и мал (7 штук).** Это осознанный размен: на
  4 ядрах перебор тысяч конфигов стоил бы минуты на каждый вызов. Широкий
  перебор — не задача этой библиотеки.
- **Замеренные GFLOP/s относятся к той машине, где сделан замер.** Тест печатает
  их как есть; никаких обещаний по производительности библиотека не даёт.

## Clean-room

Код написан с нуля по алгоритмам блочного GEMM с упаковкой панелей и
регистровым микроядром — общеизвестным приёмам численной линейной алгебры.
Ничего не скопировано из cuBLAS, NVIDIA SDK или других закрытых источников.