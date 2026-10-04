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