/*
 * Общие детали блочного GEMM: параметры вызова, выровненные буферы и упаковка
 * панелей. Объявления статические и без target-атрибутов — их можно спокойно
 * встраивать и в базовый вариант ядра, и в вариант под конкретную ISA
 * (набор инструкций вызывающего — надмножество базового).
 *
 * Определения самих ядер лежат в gemm_kernel.inc, который gemm.cpp
 * включает несколько раз: по одному разу на каждый вариант ISA.
 */
#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <new>

#include "rouge/numerics.h"

namespace rouge_numerics_detail {

#define ROUGE_CAT2(a, b) a##b
#define ROUGE_CAT(a, b) ROUGE_CAT2(a, b)

/* Всё, что нужно варианту ядра для одного вызова. Размеры/указатели живут
 * здесь, чтобы .inc не зависел от конфига и компилировался сам по себе. */
struct GemmCall {
  int m, n, k;
  const float *a; // [m][k]
  const float *b; // [k][n]
  float *c;       // [m][n]
  int tile_m, tile_n, tile_k;
  int unroll;
  int k_begin, k_end; // диапазон K у этой задачи (деление K при split_k)
  float *acc;         // [m][n] личный накопитель (уже обнулён вызывающим)
  float *pack_a[2];   // [tile_m * tile_k] x 2 — двойная буферизация по K
  float *pack_b[2];   // [tile_k * tile_n] x 2
};

/* То же для double-ядра (Wave 2: FP64 тем же блочным алгоритмом). */
struct GemmCall64 {
  int m, n, k;
  const double *a;
  const double *b;
  double *c;
  int tile_m, tile_n, tile_k;
  int unroll;
  int k_begin, k_end;
  double *acc;
  double *pack_a[2];
  double *pack_b[2];
};

/* Буфер, выровненный по 64 байтам: упакованные панели читаются и пишутся
 * векторными инструкциями, выравнивание убирает штраф за пересечение кэш-линий.
 * Шаблон по типу элемента (float по умолчанию — старые вызовы не меняются). */
template <typename T = float>
class AlignedBuffer {
public:
  AlignedBuffer() = default;
  explicit AlignedBuffer(std::size_t elems) { reset(elems); }
  ~AlignedBuffer() { std::free(ptr_); }
  AlignedBuffer(const AlignedBuffer &) = delete;
  AlignedBuffer &operator=(const AlignedBuffer &) = delete;

  void reset(std::size_t elems) {
    std::free(ptr_);
    ptr_ = nullptr;
    len_ = 0;
    if (elems == 0)
      return;
    // aligned_alloc требует размер, кратный выравниванию.
    const std::size_t bytes = ((elems * sizeof(T)) + kAlign - 1) / kAlign * kAlign;
    void *p = std::aligned_alloc(kAlign, bytes);
    if (!p)
      throw std::bad_alloc();
    ptr_ = static_cast<T *>(p);
    len_ = elems;
  }

  T *data() const { return ptr_; }

private:
  static constexpr std::size_t kAlign = 64;
  T *ptr_ = nullptr;
  std::size_t len_ = 0;
};

/* Упаковка панели A: rows x cols, результат построчный (в строке cols
 * подряд идущих элементов). Это ровно то, что нужно микроядру: A читается
 * как broadcast-скаляр a[r][k], B — как вектор. */
inline void pack_a(const float *src, int ld, int rows, int cols, float *dst) {
  for (int i = 0; i < rows; ++i) {
    const float *s = src + static_cast<std::size_t>(i) * static_cast<std::size_t>(ld);
    float *d = dst + static_cast<std::size_t>(i) * static_cast<std::size_t>(cols);
    std::memcpy(d, s, static_cast<std::size_t>(cols) * sizeof(float));
  }
}

/* Упаковка панели B: rows x cols, результат построчный. */
inline void pack_b(const float *src, int ld, int rows, int cols, float *dst) {
  for (int i = 0; i < rows; ++i) {
    const float *s = src + static_cast<std::size_t>(i) * static_cast<std::size_t>(ld);
    float *d = dst + static_cast<std::size_t>(i) * static_cast<std::size_t>(cols);
    std::memcpy(d, s, static_cast<std::size_t>(cols) * sizeof(float));
  }
}

/* Double-версии упаковщиков для GemmCall64 (та же раскладка панелей). */
inline void pack_a64(const double *src, int ld, int rows, int cols, double *dst) {
  for (int i = 0; i < rows; ++i) {
    const double *s = src + static_cast<std::size_t>(i) * static_cast<std::size_t>(ld);
    double *d = dst + static_cast<std::size_t>(i) * static_cast<std::size_t>(cols);
    std::memcpy(d, s, static_cast<std::size_t>(cols) * sizeof(double));
  }
}

inline void pack_b64(const double *src, int ld, int rows, int cols, double *dst) {
  for (int i = 0; i < rows; ++i) {
    const double *s = src + static_cast<std::size_t>(i) * static_cast<std::size_t>(ld);
    double *d = dst + static_cast<std::size_t>(i) * static_cast<std::size_t>(cols);
    std::memcpy(d, s, static_cast<std::size_t>(cols) * sizeof(double));
  }
}

/* Обнуление аккумулятора (C = A*B, а не C += A*B). */
inline void zero(float *p, std::size_t floats) {
  if (floats)
    std::memset(p, 0, floats * sizeof(float));
}

/* Сложение двух полных матриц [m][n]: c += a + b. */
inline void add_into(float *c, const float *a, const float *b, std::size_t floats) {
  for (std::size_t i = 0; i < floats; ++i)
    c[i] = a[i] + b[i];
}

/* Double-версии zero/add_into для накопителей GemmCall64. */
inline void zero64(double *p, std::size_t elems) {
  if (elems)
    std::memset(p, 0, elems * sizeof(double));
}

inline void add_into64(double *c, const double *a, const double *b, std::size_t elems) {
  for (std::size_t i = 0; i < elems; ++i)
    c[i] = a[i] + b[i];
}

} // namespace rouge_numerics_detail