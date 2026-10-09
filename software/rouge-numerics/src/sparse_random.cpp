/*
 * Sparse + random Wave 2: CSR SpMV и детерминированный RNG.
 *
 * SpMV: y = A*x, A в CSR (rowptr[nrows+1], colidx[nnz], values[nnz]).
 * Проверка границ: NULL, nrows/ncols/nnz <= 0 (nnz==0 разрешён — y=0),
 * colidx вне [0,ncols), rowptr монононичен. Нарушение -> возврат -1 без
 * записи; успех -> 0. Накопление в double с округлением в f32 в конце строки
 * (меньше порядок-зависимость, чем f32-аккумулятор).
 *
 * RNG: splitmix64 (общеизвестный, Steele et al., public domain-описание) как
 * источник u64; uniform_f32 — старшие 24 бита / 2^24 в [0,1); normal —
// Box-Muller с кэшем второй гауссианы в состоянии. Детерминирован: один seed
 * -> одна последовательность на любой платформе (только целочисленная
 * арифметика + libm для normal).
 */

#include "rouge/numerics.h"

#include <cmath>
#include <cstdint>

extern "C" {

int rouge_csr_spmv_f32(int nrows, int ncols, const int *rowptr,
                       const int *colidx, const float *values, const float *x,
                       float *y) {
  if (!rowptr || !colidx || !values || !x || !y) return -1;
  if (nrows <= 0 || ncols <= 0) return -1;
  if (rowptr[0] != 0) return -1;
  for (int i = 0; i < nrows; ++i) {
    if (rowptr[i + 1] < rowptr[i]) return -1;
  }
  const int nnz = rowptr[nrows];
  if (nnz < 0) return -1;
  for (int k = 0; k < nnz; ++k) {
    if (colidx[k] < 0 || colidx[k] >= ncols) return -1;
  }
  for (int i = 0; i < nrows; ++i) {
    double acc = 0.0;
    for (int k = rowptr[i]; k < rowptr[i + 1]; ++k)
      acc += (double)values[k] * (double)x[colidx[k]];
    y[i] = (float)acc;
  }
  return 0;
}

void rouge_rng_seed(RougeRng *rng, uint64_t seed) {
  if (!rng) return;
  rng->state = seed + 0x9e3779b97f4a7c15ull;
  rng->have_next = 0;
  rng->next_gauss = 0.0f;
  // Прогрев: splitmix требует перемешивания после init.
  for (int i = 0; i < 3; ++i) {
    uint64_t z = (rng->state += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    (void)(z ^ (z >> 31));
  }
}

uint64_t rouge_rng_next_u64(RougeRng *rng) {
  uint64_t z = (rng->state += 0x9e3779b97f4a7c15ull);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

float rouge_rng_uniform_f32(RougeRng *rng) {
  if (!rng) return 0.0f;
  const uint64_t v = rouge_rng_next_u64(rng);
  return (float)((v >> 40) & 0xffffffull) * (1.0f / 16777216.0f);
}

float rouge_rng_normal_f32(RougeRng *rng) {
  if (!rng) return 0.0f;
  if (rng->have_next) {
    rng->have_next = 0;
    return rng->next_gauss;
  }
  float u1 = 0.0f, u2 = 0.0f;
  do {
    u1 = rouge_rng_uniform_f32(rng);
  } while (u1 <= 0.0f);
  u2 = rouge_rng_uniform_f32(rng);
  const float r = std::sqrt(-2.0f * std::log(u1));
  const float t = 6.283185307179586f * u2;
  rng->next_gauss = r * std::sin(t);
  rng->have_next = 1;
  return r * std::cos(t);
}

}  // extern "C"
