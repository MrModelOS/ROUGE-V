/*
 * rouge-numerics: собственная открытая математика ROUGE — FP32 CPU-ядро GEMM
 * с автотюнером (Волна 2 роадмапа: своя математика вместо cuBLAS).
 *
 * Это НЕ обёртка над cuBLAS/BLAS/Eigen: реализация целиком своя, без внешних
 * зависимостей (только стандартная библиотека C++17). Это НЕ замена cuBLAS по
 * скорости — это FP32 CPU-ядро с измеримым на этой машине результатом.
 *
 * Соглашения (все матрицы row-major, элементы float32):
 *   A — [M][K], B — [K][N], C — [M][N], считается C = A * B.
 *   A хранится построчно (соседние элементы строки K подряд),
 *   B и C — построчно (соседние элементы строки N подряд).
 */
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Конфигурация ядра. Все поля — целые; значения <= 0 трактуются как
 * «взять умолчание» (см. rouge_gemm_default_config), так что
 * `RougeGemmConfig cfg = {0,0,0,0,0};` — корректный запрос «по умолчанию».
 *
 *   tile_m  — высота панели A (сколько строк A держим в упакованном виде)
 *   tile_n  — ширина панели B (сколько столбцов B держим в упакованном виде)
 *   tile_k  — глубина шага по K (размер одной упакованной k-панели)
 *   unroll  — во сколько раз микроядро разворачивает цикл по K (1/2/4)
 *   split_k — на сколько частей делится K; каждая часть считается своим
 *             потоком, результаты складываются. Внутри ограничивается K
 *             (для K < split_k реально частей min(split_k, K)), т.е. split_k
 *             — это верхняя граница параллелизма, а не точное число потоков.
 */
typedef struct RougeGemmConfig {
  int tile_m;
  int tile_n;
  int tile_k;
  int unroll;
  int split_k;
} RougeGemmConfig;

/* Профиль одного замера: конфиг + достигнутая производительность. */
typedef struct RougeGemmProfile {
  RougeGemmConfig cfg; /* конфиг, который реально применялся */
  double gflops;       /* 2*M*N*K / wall-clock, GFLOP/s */
  long long iters;     /* сколько раз ядро считалось во время замера */
} RougeGemmProfile;

/* Обычный блочный GEMM: C[M][N] = A[M][K] * B[K][N], с FMA, упаковкой
 * панелей и двойной буферизацией по K.
 * cfg == NULL  -> конфиг по умолчанию.
 * M/N/K <= 0   -> ничего не делать (C не трогается). */
void rouge_gemm_f32(int M, int N, int K, const float *A, const float *B, float *C,
                    const RougeGemmConfig *cfg);

/* FP64 GEMM (Wave 2): C[M][N] = A[M][K] * B[K][N] в double — тем же блочным
 * алгоритмом (упаковка, двойная буферизация, split_k, те же ISA-варианты).
 * Конфиг общий с f32. M/N/K <= 0 или NULL -> ничего не делать. */
void rouge_gemm_f64(int M, int N, int K, const double *A, const double *B,
                    double *C, const RougeGemmConfig *cfg);

/* FP8 GEMM (Wave 2): C[M][N] f32 = A[M][K] e4m3 * B[K][N] e4m3, накопление
 * в f32. Формат входов — OCP OFP8 E4M3 (bias 7, max 448): те же таблицы,
 * что в PTX-интерпретаторе и в mma.lowering (e4m3-входы mma). Реализация:
 * decode в f32-буферы + тот же блочный rouge_gemm_f32 (автотюнер shared).
 * M/N/K <= 0 или NULL-указатели -> ничего не делать. */
void rouge_gemm_e4m3(int M, int N, int K, const uint8_t *A, const uint8_t *B,
                     float *C, const RougeGemmConfig *cfg);

/* Свёртка 2D (Wave 2): Y[N][Cout][OH][OW] = conv(X[N][Cin][H][W],
 * Wt[Cout][Cin][KH][KW]) + bias[Cout], всё NCHW f32. Реализация — im2col +
 * свой блочный rouge_gemm_f32 (автотюнер shared, cfg пробрасывается).
 * OH = (H + 2*pad - KH)/stride + 1 (целочисленно). bias может быть NULL.
 * Некорректные размеры или NULL X/Wt/Y -> ничего не делать. */
void rouge_conv2d_f32(int N, int Cin, int H, int W, int Cout, int KH, int KW,
                      int stride, int pad, const float *X, const float *Wt,
                      const float *bias, float *Y,
                      const RougeGemmConfig *cfg);

/* Winograd F(2x2,3x3) для 3x3 stride 1 pad 1 (выход того же HxW).
 * Успех -> 0; неверные указатели/размеры -> -1 без записи.
 * Хвосты нечётных OH/OW досчитываются прямо внутри. */
int rouge_conv_winograd3x3_f32(int N, int Cin, int H, int W, int Cout,
                               const float *X, const float *Wt,
                               const float *bias, float *Y);

/* Профиль автотюнера свёртки: method 0=im2col, 1=winograd, 2=implicit,
 * -1=ошибка; millis — wall-clock победителя на этих данных (в Y его результат). */
typedef struct RougeConvProfile {
  int method;
  double millis;
} RougeConvProfile;

/* Автотюнер свёртки: замер im2col / Winograd (когда применим 3x3 s1p1) /
 * implicit, без кэша; в Y — результат победителя. */
RougeConvProfile rouge_conv_tuned(int N, int Cin, int H, int W, int Cout,
                                  int KH, int KW, int stride, int pad,
                                  const float *X, const float *Wt,
                                  const float *bias, float *Y);

/* Implicit-GEMM свёртка (Wave 2): тот же счёт, что im2col+GEMM, но col-буфер
 * не материализуется — тайл выхода накапливается прямо из X. tile_m/tile_n
 * из cfg задают тайлы по Cout/пространству (clamp 1..64/256, иначе 8/128).
 * Некорректные размеры или NULL -> ничего не делать. */
void rouge_conv_implicit_f32(int N, int Cin, int H, int W, int Cout, int KH,
                             int KW, int stride, int pad, const float *X,
                             const float *Wt, const float *bias, float *Y,
                             const RougeGemmConfig *cfg);

/* Backward свёртки (Wave 2, для обучения): градиенты цепным правилом по
 * определению forward. dx — по входу (транспонированная), dW — по весам.
 * Некорректные размеры или NULL -> ничего не делать. */
void rouge_conv_bwd_data(int N, int Cin, int H, int W, int Cout, int KH, int KW,
                         int stride, int pad, const float *dy, const float *Wt,
                         float *dx);
void rouge_conv_bwd_filter(int N, int Cin, int H, int W, int Cout, int KH,
                           int KW, int stride, int pad, const float *X,
                           const float *dy, float *dW);

/* Attention (Wave 2): O[S][D] = softmax(Q[s]·K/scale)·V, блочный проход по
 * K/V (Bt=64) с онлайн-пересчётом max/sum — полная S×Skv не материализуется.
 * causal!=0 маскирует t > s+kv_off; scale<=0 -> 1/sqrt(D).
 * NULL или S/Skv/D<=0 -> ничего не делать. */
void rouge_attention_f32(int S, int Skv, int D, const float *Q, const float *K,
                         const float *V, float *O, float scale, int causal,
                         int kv_off);

/* BLAS L1/L2 (Wave 2): свои axpy/dot/scal/nrm2/gemv, row-major.
 * inc <= 0, n <= 0 или NULL -> ничего не делать (dot/nrm2 возвращают 0).
 * trans для gemv: 'N'/'T' (регистр не важен), иначе ничего не делать. */
void rouge_saxpy(int n, float alpha, const float *x, int incx, float *y,
                 int incy);
void rouge_daxpy(int n, double alpha, const double *x, int incx, double *y,
                 int incy);
float rouge_sdot(int n, const float *x, int incx, const float *y, int incy);
double rouge_ddot(int n, const double *x, int incx, const double *y, int incy);
void rouge_sscal(int n, float alpha, float *x, int incx);
void rouge_dscal(int n, double alpha, double *x, int incx);
float rouge_snrm2(int n, const float *x, int incx);
double rouge_dnrm2(int n, const double *x, int incx);
void rouge_sgemv(char trans, int m, int n, float alpha, const float *A, int lda,
                 const float *x, int incx, float beta, float *y, int incy);
void rouge_dgemv(char trans, int m, int n, double alpha, const double *A,
                 int lda, const double *x, int incx, double beta, double *y,
                 int incy);

/* Комплексное число для FFT: re/im подряд. */
typedef struct RougeComplex {
  float re;
  float im;
} RougeComplex;

/* Комплексное double: re/im подряд (layout-совместимо с cuDoubleComplex). */
typedef struct RougeComplexD {
  double re;
  double im;
} RougeComplexD;

/* Комплексный GEMM (Wave 2): C = A*B поэлементно-комплексно, через 4 вещественных
 * GEMM своего ядра (Cre = Are*Bre - Aim*Bim, Cim = Are*Bim + Aim*Bre).
 * M/N/K <= 0 или NULL -> ничего не делать. */
void rouge_gemm_c32(int M, int N, int K, const RougeComplex *A,
                    const RougeComplex *B, RougeComplex *C,
                    const RougeGemmConfig *cfg);
void rouge_gemm_z64(int M, int N, int K, const RougeComplexD *A,
                    const RougeComplexD *B, RougeComplexD *C,
                    const RougeGemmConfig *cfg);

/* FFT (Wave 2): свой radix-2 Cooley-Tukey для степеней двойки; остальные
 * n >= 1 — прямым O(N^2) DFT тем же определением (медленно, но корректно).
 * inverse=0 — forward без нормировки, inverse!=0 — inverse с делением на N.
 * Успех -> 0; n <= 0 или NULL -> -1 без записи. */
int rouge_fft_f32(int n, const RougeComplex *in, RougeComplex *out,
                  int inverse);

/* CSR SpMV (Wave 2): y = A*x. rowptr[nrows+1], colidx/values[nnz].
 * Успех -> 0; нарушение границ/NULL -> -1 без записи. */
int rouge_csr_spmv_f32(int nrows, int ncols, const int *rowptr,
                       const int *colidx, const float *values, const float *x,
                       float *y);

/* Детерминированный RNG (Wave 2): splitmix64 + uniform [0,1) + normal
 * (Box-Muller с кэшем). Один seed -> одна последовательность везде. */
typedef struct RougeRng {
  uint64_t state;
  int have_next;
  float next_gauss;
} RougeRng;

void rouge_rng_seed(RougeRng *rng, uint64_t seed);
uint64_t rouge_rng_next_u64(RougeRng *rng);
float rouge_rng_uniform_f32(RougeRng *rng);
float rouge_rng_normal_f32(RougeRng *rng);

/* Квантование f32 <-> OCP E4M3 (RNE, satfinite в знаковый MAX_NORM;
 * NaN -> знаковый NaN 0x7F, Inf -> знаковый MAX 0x7E, signed zero хранится).
 * Дословная транскрипция f32_to_narrow/narrow_to_f32 интерпретатора. */
uint8_t rouge_quantize_e4m3(float f);
float rouge_dequantize_e4m3(uint8_t u);

/* FP16/BF16 (Wave 2): квантование RNE; f16 — IEEE 754 half (переполнение в
 * Inf), bf16 — округление старших 16 бит (переполнение в Inf).
 * NaN -> канонический NaN, signed zero хранится. */
uint16_t rouge_quantize_f16(float f);
float rouge_dequantize_f16(uint16_t u);
uint16_t rouge_quantize_bf16(float f);
float rouge_dequantize_bf16(uint16_t u);

/* GEMM на половинных входах: C[M][N] f32 = A[M][K] * B[K][N], накопление f32.
 * Реализация — decode в f32 + тот же блочный rouge_gemm_f32 (как e4m3).
 * M/N/K <= 0 или NULL -> ничего не делать. */
void rouge_gemm_f16(int M, int N, int K, const uint16_t *A, const uint16_t *B,
                    float *C, const RougeGemmConfig *cfg);
void rouge_gemm_bf16(int M, int N, int K, const uint16_t *A, const uint16_t *B,
                     float *C, const RougeGemmConfig *cfg);

/* Конфиг по умолчанию (умеренный: панели 64x64x32, unroll 2, split_k 1). */
RougeGemmConfig rouge_gemm_default_config(void);

/* Набор кандидатов для автотюнера. Ровно столько конфигов, сколько вернёт
 * rouge_gemm_candidate_count(); тесты прогоняют корректность по каждому.
 * index вне диапазона -> возвращается конфиг по умолчанию. */
int rouge_gemm_candidate_count(void);
RougeGemmConfig rouge_gemm_candidate(int index);

/* Скорость одного конфига на этих данных: 2*M*N*K / лучшее wall-clock из
 * нескольких прогонов, GFLOP/s. Замер честный — считает на настоящих
 * данных A/B/C, результат пишется в C. cfg == NULL -> по умолчанию. */
double rouge_gemm_flops_per_sec(int M, int N, int K, const float *A, const float *B, float *C,
                                const RougeGemmConfig *cfg);

/* Автотюнер: перебирает rouge_gemm_candidate(), замеряет wall-clock каждого
 * на этих данных, оставляет в C результат лучшего конфига и возвращает
 * профиль. Кэша нет: каждый прогон считает по-настоящему. */
RougeGemmProfile rouge_gemm_tuned(int M, int N, int K, const float *A, const float *B, float *C);

/* Имя ISA-варианта, которым реально считает этот процессор
 * ("base+sse2", "avx2+fma", "avx512+fma"); для отчётов о замерах.
 * Переменная окружения ROUGE_NUMERICS_ISA ("base", "avx2", "avx512")
 * принудительно выбирает вариант — ею проверяют тесты все собранные пути;
 * если процессор не умеет запрошенный вариант, просьба отклоняется. */
const char *rouge_gemm_isa_variant(void);

#ifdef __cplusplus
} /* extern "C" */
#endif