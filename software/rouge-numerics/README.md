# rouge-numerics — своя открытая математика вместо cuBLAS

**Что это.** Своя открытая математика ROUGE-V (Волна 2 роадмапа): блочные
FP32/FP64 CPU-GEMM с автотюнером, GEMM на половинных/FP8/комплексных входах,
свёртка (im2col+GEMM, implicit без col-буфера, Winograd F(2x2,3x3), автотюнер
метода), блочный attention с онлайн-softmax, свой FFT (быстрый radix-2 +
прямой путь для произвольных n), CSR SpMV, детерминированный RNG и шимы
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
- Шимы покрывают **BLAS L1/L2/L3 (S/D/C/Z, strided-batched) и рабочую базу
  cuDNN (conv-forward/backward, активации, AddTensor, Softmax,
  BatchNorm-inference)**; остального (LT, Xt, RNN, обучение вне conv)
  нет намеренно — громкая ошибка линковки вместо тихого вранья.
- FFT не-степеней двойки — прямым O(N²) путём (медленно, корректно).

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
| `gemm_f64` (+ `gemm_f64_isa_*`) | FP64: каждый кандидат на размерах с краями против наивного эталона в long double (допуск 1e-12), split_k, ISA-матрица как для f32 |
| `complex` | c32/z64 против эталона в long double (допуски 1e-6/1e-14) |
| `conv` | Свёртка NCHW: ручной случай 3x3 (−6) плюс 6 случайных конфигураций (stride/pad/bias, неквадратные ядра) против независимого прямого эталона |
| `implicit` | Implicit-свёртка (без col-буфера) на 6 конфигурациях против im2col (допуск 1e-5) + тюнер трёх методов везде возвращает 0/1/2 с верным результатом |
| `winograd` | Winograd F(2x2,3x3): ручной случай (единичное ядро — выход равен входу) + 5 случайных конфигураций (чёт/нечет, bias) против im2col, допуск 1e-4; плюс автотюнер метода |
| `attention` | Блочный attention против наивного двухпроходного эталона в double: обычный, causal, causal+сдвиг, Skv=130 (пересчёт m/l через границу блока), допуск 1e-5 |
| `fft_sparse_rng` | FFT n=1..256 против наивного DFT + roundtrip; произвольные n=3..127 против long double-эталона + roundtrip; SpMV на случайной разреженности против плотного эталона бит-в-бит + проверки границ; RNG: детерминизм, uniform в [0,1), normal mean/var на 100k |
| `bench_fftw` | Наш FFT против FFTW (single, ESTIMATE): корректность 1e-4 на n=64..4096 и 100/1000 + таблица времён; см. «Замеры» |
| `shims` | `cublasSgemm` все 4 комбинации N/T против эталона в double; `Dgemm` N/N+T/T; `Cgemm` с сопряжением и `Zgemm` против long double; strided-batched S/D; L1/L2 (axpy/copy/swap/scal/dot(u/c)/nrm2/gemv, pointer mode); `cudnnConvolutionForward` оба алгоритма; AddTensor/активации/Softmax/BatchNorm/conv-backward против прямых формул + отказы |
| `bench_blas` | Замер нашего GEMM против системного BLAS (OpenBLAS предпочтительно) и наивной петли; корректность обязательна, скорость — отчёт; OpenBLAS меряется и в 1 поток (наше ядро однопоточное), и по умолчанию; см. «Замеры» |
| `bench_dnnl` | Свёртка против oneDNN (convolution_auto) на 7 формах: корректность ≤1e-6, отношение времён; см. «Замеры» |
| `blas1` | L1/L2 против прямых петель + против системного cblas (axpy/dot/scal/nrm2/gemv, inc≠1, отказ на мусорном trans) |
| `bwd` | Backward свёртки против численного градиента (центральные разности): dx и dw на 1x1x4x4/3x3, dx на stride-2+pad |

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

/* FP64 GEMM тем же блочным алгоритмом (те же ISA-варианты, NR вдвое уже). */
void rouge_gemm_f64(int M, int N, int K, const double* A, const double* B,
                    double* C, const RougeGemmConfig* cfg);

/* Комплексный GEMM через 4 вещественных (Cre = Are*Bre − Aim*Bim, ...). */
void rouge_gemm_c32(int M, int N, int K, const RougeComplex* A,
                    const RougeComplex* B, RougeComplex* C,
                    const RougeGemmConfig* cfg);
void rouge_gemm_z64(int M, int N, int K, const RougeComplexD* A,
                    const RougeComplexD* B, RougeComplexD* C,
                    const RougeGemmConfig* cfg);

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

/* Implicit-свёртка: тот же счёт без col-буфера (память O(tc*ts)).
 * tile_m/tile_n из cfg — тайлы по Cout/пространству, иначе 8/128. */
void rouge_conv_implicit_f32(int N, int Cin, int H, int W, int Cout, int KH,
                             int KW, int stride, int pad, const float* X,
                             const float* Wt, const float* bias, float* Y,
                             const RougeGemmConfig* cfg);

/* Winograd F(2x2,3x3) для 3x3 stride 1 pad 1. Успех -> 0, иначе -1. */
int rouge_conv_winograd3x3_f32(int N, int Cin, int H, int W, int Cout,
                               const float* X, const float* Wt,
                               const float* bias, float* Y);

typedef struct RougeConvProfile { int method; double millis; } RougeConvProfile;
/* Автотюнер свёртки: замер im2col/Winograd/implicit; в Y результат победителя.
 * method: 0=im2col, 1=winograd, 2=implicit, -1=ошибка. */
RougeConvProfile rouge_conv_tuned(int N, int Cin, int H, int W, int Cout,
                                  int KH, int KW, int stride, int pad,
                                  const float* X, const float* Wt,
                                  const float* bias, float* Y);

/* Backward свёртки (обучение): dx по входу, dW по весам. */
void rouge_conv_bwd_data(int N, int Cin, int H, int W, int Cout, int KH, int KW,
                         int stride, int pad, const float* dy, const float* Wt,
                         float* dx);
void rouge_conv_bwd_filter(int N, int Cin, int H, int W, int Cout, int KH,
                           int KW, int stride, int pad, const float* X,
                           const float* dy, float* dW);

/* BLAS L1/L2, row-major. inc<=0/n<=0/NULL -> no-op (dot/nrm2 дают 0). */
void rouge_saxpy(int n, float alpha, const float* x, int incx, float* y, int incy);
void rouge_daxpy(int n, double alpha, const double* x, int incx, double* y, int incy);
float rouge_sdot(int n, const float* x, int incx, const float* y, int incy);
double rouge_ddot(int n, const double* x, int incx, const double* y, int incy);
void rouge_sscal(int n, float alpha, float* x, int incx);
void rouge_dscal(int n, double alpha, double* x, int incx);
float rouge_snrm2(int n, const float* x, int incx);
double rouge_dnrm2(int n, const double* x, int incx);
void rouge_sgemv(char trans, int m, int n, float alpha, const float* A, int lda,
                 const float* x, int incx, float beta, float* y, int incy);
void rouge_dgemv(char trans, int m, int n, double alpha, const double* A, int lda,
                 const double* x, int incx, double beta, double* y, int incy);

/* Attention: O[S][D] = softmax(Q·K/scale)·V, блочный проход Bt=64.
 * causal маскирует t > s+kv_off; scale<=0 -> 1/sqrt(D). */
void rouge_attention_f32(int S, int Skv, int D, const float* Q, const float* K,
                         const float* V, float* O, float scale, int causal,
                         int kv_off);

typedef struct RougeComplex { float re, im; } RougeComplex;
/* FFT: степени двойки — radix-2, остальные n — прямой O(N²) тем же
 * определением. inverse=0 forward, иначе inverse с /N. Успех -> 0. */
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
               B, ldb, &beta, C, ldc);   /* column-major; S/D/C/Z + batched S/D */

#include <rouge/cudnn_shim.h>
cudnnCreate(&h); /* дескрипторы NCHW/FLOAT: conv-forward/backward, активации,
                    AddTensor, Softmax, BatchNorm-inference */
```

## Замеры (i5-1135G7, путь `avx512+fma`, минимум из 5–7 прогонов wall-clock)

### GEMM против OpenBLAS 0.3.34 (Arch extra)

`bench_blas`. Наше ядро **однопоточное** (параллелизм только по `split_k`),
поэтому основное сравнение — с OpenBLAS, ограниченным одним потоком
(`openblas_set_num_threads(1)`): это «ядро против ядра». Второй столбец —
OpenBLAS в режиме по умолчанию (все ядра), он на этой машине нестабилен
(на 128×128×128 между прогонами гуляло 1.4 → 181 GFLOP/s), поэтому в вывод
не лезет.

| Размер | rouge_gemm | OpenBLAS (1 поток) | OpenBLAS (по умолчанию) | наивная петля |
| --- | --- | --- | --- | --- |
| f32 64×64×64 | ~84 GFLOP/s | ~117 | ~118 | ~2.4 |
| f32 128×128×128 | ~88 GFLOP/s | ~104 | 1.4…181 (нестабильно) | ~2.0 |
| f32 256×192×128 | ~78 GFLOP/s | ~112 | ~263–351 | ~2.2 |
| f64 128×128×128 | ~40 GFLOP/s | ~50 | ~137–157 | ~1.9 |

Честный вывод: в однопоточном режиме наше ядро **проигрывает OpenBLAS в
1.2–1.3 раза** — это честная граница, а не победа. Многопоточность OpenBLAS
(×8 потоков) на больших панелях даёт ×3–4 к нашему одному потоку; наш
`split_k` на этой машине даёт лишь несколько процентов (см. «Узкие места»).
Выводы «мы быстрее OpenBLAS» не делаем.

### Свёртка против oneDNN 3.11.3 (Arch extra)

`bench_dnnl`. Одинаковые данные NCHW, одна математика (cross-correlation),
oneDNN в режиме `convolution_auto` (сам выбирает реализацию — иначе сравнение
было бы нечестным). Корректность: расхождение с oneDNN ≤ 1e-6.

| Случай | rouge | oneDNN | отношение |
| --- | --- | --- | --- |
| N=1, 16→16, 32×32, 3×3 s1 p1 | 0.21 мс | 3.11 мс | ~15× |
| N=2, 32→32, 32×32, 3×3 s1 p1 | 1.08 мс | 21.9 мс | ~20× |
| N=1, 16→32, 28×28, 3×3 s1 p1 | 0.23 мс | 4.67 мс | ~21× |
| N=2, 8→16, 16×16, 5×5 s2 p2 | 0.04 мс | 0.42 мс | ~12× |
| N=1, 64→64, 56×56, 3×3 s1 p1 | 4.53 мс | 117.7 мс | ~26× |
| N=4, 32→64, 32×32, 3×3 s1 p1 | 3.07 мс | 87.2 мс | ~28× |
| N=1, 3→64, 224×224, 7×7 s2 p3 | 4.55 мс | 194.8 мс | ~43× |

Оговорки, без которых цифра вводит в заблуждение: сравнивается **стоковый
пакет** oneDNN без ручной настройки (его brgemm-JIT мы не включали и не
тюнили) против нашего пути im2col+свой GEMM, у которого есть автотюнер; на
малых формах у oneDNN заметная доля времени — накладные расходы на вызов.
Проверялось на тех же данных, в тех же процессах; корректность обязательна
(иначе тест падает).

### FFT против FFTW 3.7.11

`bench_fftw`, single precision, `FFTW_ESTIMATE`, корректность 1e-4 везде:

| n | rouge | FFTW | комментарий |
| --- | --- | --- | --- |
| 64 | 0.001 мс | ~0.000 мс | на пределе разрешения таймера |
| 256 | 0.003 мс | ~0.000 мс | то же |
| 1024 | 0.016 мс | 0.001 мс | ~16×: у FFTW кодлеты+SIMD+планирование |
| 4096 | 0.077 мс | 0.006 мс | ~13×, та же причина |
| 100 | 0.157 мс | ~0.000 мс | наш O(N²)-путь против O(N log N) |
| 1000 | 14.172 мс | 0.002 мс | ~7000×: медленный путь честно медленный |

Вывод без прикрас: наш FFT корректен, но не конкурент FFTW по скорости —
скалярный radix-2 против кодлетов с SIMD. Медленный путь для не-степеней
двойки — для корректности там, где размер неудобный, а не для скорости.

### Своя свёртка: три метода

| Случай | im2col | Winograd | implicit | выбор тюнера |
| --- | --- | --- | --- | --- |
| 16→16, 32×32 | 0.33 мс | 0.36 мс (отн. 2e-7) | 3.47 мс (отн. 4e-7) | im2col |
| 32→32, 32×32 | 0.91 мс | 1.21 мс (отн. 3e-7) | 13.68 мс (отн. 7e-7) | im2col |

Всё корректно (<1e-6 друг к другу), но на этом CPU побеждает im2col+AVX512-GEMM:
поточечные циклы Winograd/implicit скалярные, а микроядро GEMM векторизовано.
Implicit при этом не держит col-буфер (память O(tc·ts) против O(K·Ncol) —
на больших картах это разница между «влезло» и «нет»), так что это размен
«память против скорости», а не проигрыш. Автотюнер `rouge_conv_tuned` для
того и нужен — он выбирает победителя замером, а не верой.

Как подключить эталоны: `-DROUGE_OPENBLAS_ROOT=<префикс>` и
`-DROUGE_ONEDNN_ROOT=<префикс>` (в префиксе `lib/` и `include/`); без них
тесты собираются с системным CBLAS или вообще без него, и в выводе печатается,
какая именно библиотека сравнивалась.

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

**FP64 тем же файлом.** `gemm_kernel.inc` параметризован макросами
`ROUGE_KERNEL_ELEM/CALL/PACK_A/PACK_B` (по умолчанию float): `gemm.cpp`
включает его ещё три раза для double (`dbase`/`davx2`/`davx512`, NR вдвое
уже — регистры те же, элементы шире). Диспетчер ISA и нормализация конфига
общие с f32. Комплексные GEMM (`src/complex_gemm.cpp`) отдельного микроядра
не имеют — четыре вещественных GEMM своего ядра.

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