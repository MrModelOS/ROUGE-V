/*
 * BLAS L1/L2 Wave 2: свои axpy/dot/scal/nrm2/gemv поверх тех же приёмов, что
 * GEMM (блокирование, автотюнеру здесь делать нечего — операции memory-bound).
 *
 * Соглашения: row-major, incx/incy — шаги (<=0 и n<=0 -> ничего не делать,
 * для dot/nrm2 результат 0.0); trans для gemv: 'N'/'T' (регистр не важен),
 * иначе ничего не делать. NaN/Inf арифметика — обычная IEEE хоста.
 * gemv считает y = alpha*op(A)*x + beta*y с блокированием по строкам
 * (панель 64) для кэш-дружелюбия; dot — два накопителя (попарное суммирование
 * вместо наивной цепочки — меньше ошибка, тест это проверяет допуском).
 */

#include "rouge/numerics.h"

#include <cmath>
#include <cctype>

extern "C" {

void rouge_saxpy(int n, float alpha, const float *x, int incx, float *y,
                 int incy) {
  if (!x || !y || n <= 0 || incx <= 0 || incy <= 0) return;
  for (int i = 0; i < n; ++i)
    y[(size_t)i * incy] += alpha * x[(size_t)i * incx];
}

void rouge_daxpy(int n, double alpha, const double *x, int incx, double *y,
                 int incy) {
  if (!x || !y || n <= 0 || incx <= 0 || incy <= 0) return;
  for (int i = 0; i < n; ++i)
    y[(size_t)i * incy] += alpha * x[(size_t)i * incx];
}

float rouge_sdot(int n, const float *x, int incx, const float *y, int incy) {
  if (!x || !y || n <= 0 || incx <= 0 || incy <= 0) return 0.0f;
  double s0 = 0.0, s1 = 0.0;  // два накопителя: чётные/нечётные
  for (int i = 0; i < n; ++i) {
    const double p =
        (double)x[(size_t)i * incx] * (double)y[(size_t)i * incy];
    if (i & 1)
      s1 += p;
    else
      s0 += p;
  }
  return (float)(s0 + s1);
}

double rouge_ddot(int n, const double *x, int incx, const double *y,
                  int incy) {
  if (!x || !y || n <= 0 || incx <= 0 || incy <= 0) return 0.0;
  double s0 = 0.0, s1 = 0.0;
  for (int i = 0; i < n; ++i) {
    const double p = x[(size_t)i * incx] * y[(size_t)i * incy];
    if (i & 1)
      s1 += p;
    else
      s0 += p;
  }
  return s0 + s1;
}

void rouge_sscal(int n, float alpha, float *x, int incx) {
  if (!x || n <= 0 || incx <= 0) return;
  for (int i = 0; i < n; ++i) x[(size_t)i * incx] *= alpha;
}

void rouge_dscal(int n, double alpha, double *x, int incx) {
  if (!x || n <= 0 || incx <= 0) return;
  for (int i = 0; i < n; ++i) x[(size_t)i * incx] *= alpha;
}

float rouge_snrm2(int n, const float *x, int incx) {
  if (!x || n <= 0 || incx <= 0) return 0.0f;
  // Масштабированное суммирование квадратов (приём из reference BLAS):
  // переполнение/недобор промежуточных квадратов не ломают результат.
  double scale = 0.0, ssq = 1.0;
  for (int i = 0; i < n; ++i) {
    const double v = std::fabs((double)x[(size_t)i * incx]);
    if (v != 0.0) {
      if (scale < v) {
        ssq = 1.0 + ssq * (scale / v) * (scale / v);
        scale = v;
      } else {
        ssq += (v / scale) * (v / scale);
      }
    }
  }
  return (float)(scale * std::sqrt(ssq));
}

double rouge_dnrm2(int n, const double *x, int incx) {
  if (!x || n <= 0 || incx <= 0) return 0.0;
  double scale = 0.0, ssq = 1.0;
  for (int i = 0; i < n; ++i) {
    const double v = std::fabs(x[(size_t)i * incx]);
    if (v != 0.0) {
      if (scale < v) {
        ssq = 1.0 + ssq * (scale / v) * (scale / v);
        scale = v;
      } else {
        ssq += (v / scale) * (v / scale);
      }
    }
  }
  return scale * std::sqrt(ssq);
}

void rouge_sgemv(char trans, int m, int n, float alpha, const float *A, int lda,
                 const float *x, int incx, float beta, float *y, int incy) {
  if (!A || !x || !y || m <= 0 || n <= 0 || lda <= 0 || incx <= 0 || incy <= 0)
    return;
  const char t = (char)std::toupper((unsigned char)trans);
  if (t != 'N' && t != 'T') return;
  constexpr int kPanel = 64;
  if (t == 'N') {
    // y[m] = alpha*A[m][n]*x + beta*y, A row-major с шагом lda.
    for (int i0 = 0; i0 < m; i0 += kPanel) {
      const int i1 = i0 + kPanel < m ? i0 + kPanel : m;
      for (int i = i0; i < i1; ++i) {
        double s = 0.0;
        const float *row = A + (size_t)i * lda;
        for (int j = 0; j < n; ++j) s += (double)row[j] * x[(size_t)j * incx];
        y[(size_t)i * incy] = (float)(alpha * s + beta * y[(size_t)i * incy]);
      }
    }
  } else {
    // y[n] = alpha*A^T*x + beta*y.
    for (int j = 0; j < n; ++j)
      y[(size_t)j * incy] = (float)(beta * y[(size_t)j * incy]);
    for (int i0 = 0; i0 < m; i0 += kPanel) {
      const int i1 = i0 + kPanel < m ? i0 + kPanel : m;
      for (int i = i0; i < i1; ++i) {
        const float xv = (float)(alpha * x[(size_t)i * incx]);
        const float *row = A + (size_t)i * lda;
        for (int j = 0; j < n; ++j)
          y[(size_t)j * incy] += xv * row[j];
      }
    }
  }
}

void rouge_dgemv(char trans, int m, int n, double alpha, const double *A,
                 int lda, const double *x, int incx, double beta, double *y,
                 int incy) {
  if (!A || !x || !y || m <= 0 || n <= 0 || lda <= 0 || incx <= 0 || incy <= 0)
    return;
  const char t = (char)std::toupper((unsigned char)trans);
  if (t != 'N' && t != 'T') return;
  constexpr int kPanel = 64;
  if (t == 'N') {
    for (int i0 = 0; i0 < m; i0 += kPanel) {
      const int i1 = i0 + kPanel < m ? i0 + kPanel : m;
      for (int i = i0; i < i1; ++i) {
        long double s = 0.0L;
        const double *row = A + (size_t)i * lda;
        for (int j = 0; j < n; ++j)
          s += (long double)row[j] * x[(size_t)j * incx];
        y[(size_t)i * incy] = alpha * (double)s + beta * y[(size_t)i * incy];
      }
    }
  } else {
    for (int j = 0; j < n; ++j) y[(size_t)j * incy] = beta * y[(size_t)j * incy];
    for (int i0 = 0; i0 < m; i0 += kPanel) {
      const int i1 = i0 + kPanel < m ? i0 + kPanel : m;
      for (int i = i0; i < i1; ++i) {
        const double xv = alpha * x[(size_t)i * incx];
        const double *row = A + (size_t)i * lda;
        for (int j = 0; j < n; ++j) y[(size_t)j * incy] += xv * row[j];
      }
    }
  }
}

}  // extern "C"
