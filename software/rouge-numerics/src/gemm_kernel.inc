/*
 * Тело варианта блочного GEMM-ядра. Файл НЕ компилируется сам по себе: его
 * несколько раз включает gemm.cpp — по одному разу на каждый вариант ISA
 * (базовый, avx2+fma, avx512f+fma). Один и тот же алгоритм собирается под
 * разные наборы инструкций: сравниваются только замеры, а не разные алгоритмы.
 *
 * Что определяет включающий файл (макросы):
 *   ROUGE_KERNEL_NAME    уникальный суффикс имён функций этого варианта
 *   ROUGE_KERNEL_TARGET  target-атрибут варианта (пусто для базовой ISA)
 *   ROUGE_KERNEL_MR      строк в регистровом блоке микроядра
 *   ROUGE_KERNEL_NR      столбцов в регистровом блоке микроядра
 *
 * Алгоритм (всё row-major: A[M][K], B[K][N], считается C = A*B):
 *   1. панели A (tile_m x tile_k) и B (tile_k x tile_n) упаковываются так,
 *      что микроядру удобно: A[i][k] — broadcast-скаляр, B[k][j] — вектор;
 *   2. двойная буферизация по K: следующая k-панель упаковывается в запасной
 *      буфер до вычисления текущей, поэтому данные, которые ещё читаются,
 *      не затираются;
 *   3. микроядро MR x NR держит полосы C в регистрах, цикл по K
 *      разворачивается в UNROLL шагов, плитка идёт блоками MR x NR;
 *   4. края (mrows/ncols, не кратные плитке) считает edge-ядро — там границы
 *      известны только в рантайме, полосы в регистрах не удержать.
 *
 * ВАЖНО про шаги: упакованная панель A имеет шаг klen по строкам, панель B —
 * шаг ncols (ширина всей плитки), а не NR. Поэтому шаг передаётся параметром.
 */

/* Упаковка одной k-панели. */
#define ROUGE_PACK_PANEL ROUGE_CAT(pack_panel_, ROUGE_KERNEL_NAME)
/* Полный блок MR x NR микроядра. */
#define ROUGE_MICRO ROUGE_CAT(micro_, ROUGE_KERNEL_NAME)
/* Краевой блок (произвольные mr x nr, где nr <= NR). */
#define ROUGE_EDGE ROUGE_CAT(edge_, ROUGE_KERNEL_NAME)
/* Вычисление одной панели сеткой блоков MR x NR. */
#define ROUGE_PANEL ROUGE_CAT(panel_, ROUGE_KERNEL_NAME)
/* Драйвер варианта: тайлинг M/N + двойная буферизация по K. */
#define ROUGE_BLOCKED ROUGE_CAT(blocked_, ROUGE_KERNEL_NAME)
/* Подсказки загрузке для варианта. */
#define ROUGE_PREFETCH ROUGE_CAT(prefetch_, ROUGE_KERNEL_NAME)
/* Точка входа варианта: разводит конфиг по шаблонам времени компиляции. */
#define ROUGE_RUN ROUGE_CAT(run_, ROUGE_KERNEL_NAME)

/* Тип элемента и окружение вызова. По умолчанию float/GemmCall (gemm.cpp ничего
 * не меняет); gemm_f64-варианты включают файл с ROUGE_KERNEL_ELEM=double,
 * ROUGE_KERNEL_CALL=GemmCall64 и своими упаковщиками. */
#ifndef ROUGE_KERNEL_ELEM
#define ROUGE_KERNEL_ELEM float
#endif
#ifndef ROUGE_KERNEL_CALL
#define ROUGE_KERNEL_CALL GemmCall
#endif
#ifndef ROUGE_KERNEL_PACK_A
#define ROUGE_KERNEL_PACK_A pack_a
#endif
#ifndef ROUGE_KERNEL_PACK_B
#define ROUGE_KERNEL_PACK_B pack_b
#endif

namespace rouge_numerics_detail {

/* Микроядро полного блока MR x NR: все индексы известны на этапе компиляции,
 * поэтому полосы C живут в регистрах. */
template <int MR, int NR, int UNROLL>
ROUGE_KERNEL_TARGET static void ROUGE_MICRO(const ROUGE_KERNEL_ELEM *a, int lda, const ROUGE_KERNEL_ELEM *b, int ldb,
                                            ROUGE_KERNEL_ELEM *c, int ldc, int kc) {
  ROUGE_KERNEL_ELEM acc[MR][NR];
  for (int r = 0; r < MR; ++r)
    for (int j = 0; j < NR; ++j)
      acc[r][j] = 0.0f;

  // Внутренний цикл по K разворачивается ровно в UNROLL шагов; хвост
  // (klen, не кратный UNROLL) дорабатывается последней короткой итерацией.
  for (int k = 0; k < kc;) {
    const int kend = (UNROLL <= kc - k) ? k + UNROLL : kc;
    for (int kk = k; kk < kend; ++kk) {
      const ROUGE_KERNEL_ELEM *b_row = b + static_cast<std::size_t>(kk) * ldb;
      for (int r = 0; r < MR; ++r) {
        const ROUGE_KERNEL_ELEM av = a[static_cast<std::size_t>(r) * lda + kk];
        for (int j = 0; j < NR; ++j)
          acc[r][j] += av * b_row[j];
      }
    }
    k = kend;
  }

  for (int r = 0; r < MR; ++r) {
    ROUGE_KERNEL_ELEM *c_row = c + static_cast<std::size_t>(r) * ldc;
    for (int j = 0; j < NR; ++j)
      c_row[j] += acc[r][j];
  }
}

/* Краевой блок: те же операции, но mr/nr приходят из рантайма, поэтому полосы C
 * держим в небольшом локальном буфере, а не в регистрах. Вызывающий гарантирует
 * mr <= MR и nr <= NR (край — это то, что не поместилось в полный блок). */
template <int UNROLL>
ROUGE_KERNEL_TARGET static void ROUGE_EDGE(const ROUGE_KERNEL_ELEM *a, int lda, const ROUGE_KERNEL_ELEM *b, int ldb,
                                           ROUGE_KERNEL_ELEM *c, int ldc, int mr, int nr, int kc) {
  ROUGE_KERNEL_ELEM tmp[ROUGE_KERNEL_NR];
  for (int r = 0; r < mr; ++r) {
    for (int j = 0; j < nr; ++j)
      tmp[j] = 0.0f;
    const ROUGE_KERNEL_ELEM *a_row = a + static_cast<std::size_t>(r) * lda;
    for (int k = 0; k < kc;) {
      const int kend = (UNROLL <= kc - k) ? k + UNROLL : kc;
      for (int kk = k; kk < kend; ++kk) {
        const ROUGE_KERNEL_ELEM av = a_row[kk];
        const ROUGE_KERNEL_ELEM *b_row = b + static_cast<std::size_t>(kk) * ldb;
        for (int j = 0; j < nr; ++j)
          tmp[j] += av * b_row[j];
      }
      k = kend;
    }
    ROUGE_KERNEL_ELEM *c_row = c + static_cast<std::size_t>(r) * ldc;
    for (int j = 0; j < nr; ++j)
      c_row[j] += tmp[j];
  }
}

/* Плитка (mrows x ncols) одной k-панели: идём микро-панельками по N шириной NR
 * (крайняя может быть уже), внутри — блоки по M высотой MR. */
template <int MR, int NR, int UNROLL>
ROUGE_KERNEL_TARGET static void ROUGE_PANEL(const ROUGE_KERNEL_ELEM *ap, int klen, int mrows, int ncols,
                                            const ROUGE_KERNEL_ELEM *bp, ROUGE_KERNEL_ELEM *c, int ldc) {
  for (int j = 0; j < ncols; j += NR) {
    const int nr = (ncols - j < NR) ? (ncols - j) : NR;
    const ROUGE_KERNEL_ELEM *b_ptr = bp + j;
    int i = 0;
    for (; i + MR <= mrows; i += MR) {
      const ROUGE_KERNEL_ELEM *a_ptr = ap + static_cast<std::size_t>(i) * klen;
      ROUGE_KERNEL_ELEM *c_ptr = c + static_cast<std::size_t>(i) * ldc + j;
      if (nr == NR) {
        ROUGE_MICRO<MR, NR, UNROLL>(a_ptr, klen, b_ptr, ncols, c_ptr, ldc, klen);
      } else {
        ROUGE_EDGE<UNROLL>(a_ptr, klen, b_ptr, ncols, c_ptr, ldc, MR, nr, klen);
      }
    }
    if (i < mrows) {
      // Хвост по M: строк меньше MR, поэтому полосы в регистрах не удержать.
      const ROUGE_KERNEL_ELEM *a_ptr = ap + static_cast<std::size_t>(i) * klen;
      ROUGE_KERNEL_ELEM *c_ptr = c + static_cast<std::size_t>(i) * ldc + j;
      ROUGE_EDGE<UNROLL>(a_ptr, klen, b_ptr, ncols, c_ptr, ldc, mrows - i, nr, klen);
    }
  }
}

/* Упаковка панелей A и B для строк [m0, m0+mrows), столбцов [n0, n0+ncols)
 * и k-панели [k0, k0+klen). Шаги в источниках: A — K, B — N. */
ROUGE_KERNEL_TARGET static void ROUGE_PACK_PANEL(const ROUGE_KERNEL_CALL &call, int m0, int mrows, int n0,
                                                 int ncols, int k0, int klen, ROUGE_KERNEL_ELEM *ap,
                                                 ROUGE_KERNEL_ELEM *bp) {
  ROUGE_KERNEL_PACK_A(call.a + static_cast<std::size_t>(m0) * call.k + k0, call.k, mrows, klen, ap);
  ROUGE_KERNEL_PACK_B(call.b + static_cast<std::size_t>(k0) * call.n + n0, call.n, klen, ncols, bp);
}

#if defined(__GNUC__) || defined(__clang__)
/* Подсказка загрузке для начала следующей упакованной панели.
 * __builtin_prefetch не требует ни intrinsics, ни target-атрибута. */
ROUGE_KERNEL_TARGET static void ROUGE_PREFETCH(const ROUGE_KERNEL_ELEM *p, int lines, int stride) {
  for (int i = 0; i < lines; ++i) {
    __builtin_prefetch(p + static_cast<std::size_t>(i) * stride, 0, 1);
  }
}
#else
ROUGE_KERNEL_TARGET static void ROUGE_PREFETCH(const ROUGE_KERNEL_ELEM * /*p*/, int /*lines*/, int /*stride*/) {
}
#endif

/* Драйвер варианта: тайлинг по M/N плюс двойная буферизация по K.
 * Складывает A*B в накопитель call.acc, который обнулил вызывающий. */
template <int MR, int NR, int UNROLL>
ROUGE_KERNEL_TARGET static void ROUGE_BLOCKED(const ROUGE_KERNEL_CALL &call) {
  const int tm = call.tile_m, tn = call.tile_n, tk = call.tile_k;
  ROUGE_KERNEL_ELEM *const acc = call.acc;
  if (call.k_begin >= call.k_end)
    return;

  for (int m0 = 0; m0 < call.m; m0 += tm) {
    const int mrows = (call.m - m0 < tm) ? (call.m - m0) : tm;
    for (int n0 = 0; n0 < call.n; n0 += tn) {
      const int ncols = (call.n - n0 < tn) ? (call.n - n0) : tn;
      ROUGE_KERNEL_ELEM *c_ptr = acc + static_cast<std::size_t>(m0) * call.n + n0;

      ROUGE_KERNEL_ELEM *ap[2] = {call.pack_a[0], call.pack_a[1]};
      ROUGE_KERNEL_ELEM *bp[2] = {call.pack_b[0], call.pack_b[1]};
      int cur = 0;
      int k0 = call.k_begin;
      int klen = (call.k_end - k0 < tk) ? (call.k_end - k0) : tk;
      ROUGE_PACK_PANEL(call, m0, mrows, n0, ncols, k0, klen, ap[cur], bp[cur]);

      for (;;) {
        const int nxt = 1 - cur;
        const int k0n = k0 + klen;
        int klen_next = 0;
        if (k0n < call.k_end) {
          klen_next = (call.k_end - k0n < tk) ? (call.k_end - k0n) : tk;
          // Запасной слот: текущая панель, которую мы сейчас читаем, не затирается.
          ROUGE_PACK_PANEL(call, m0, mrows, n0, ncols, k0n, klen_next, ap[nxt], bp[nxt]);
          ROUGE_PREFETCH(ap[nxt], (mrows < 4) ? mrows : 4, klen_next);
          ROUGE_PREFETCH(bp[nxt], (klen_next < 4) ? klen_next : 4, ncols);
        }
        ROUGE_PANEL<MR, NR, UNROLL>(ap[cur], klen, mrows, ncols, bp[cur], c_ptr, call.n);
        if (klen_next == 0)
          break;
        k0 = k0n;
        klen = klen_next;
        cur = nxt;
      }
    }
  }
}

} // namespace rouge_numerics_detail

/* Точка входа варианта: разводит поле unroll по шаблонам времени компиляции,
 * чтобы разворот цикла по K был настоящим, а не «как получится». */
ROUGE_KERNEL_TARGET static void ROUGE_RUN(const rouge_numerics_detail::ROUGE_KERNEL_CALL &call) {
  switch (call.unroll) {
  case 1:
    rouge_numerics_detail::ROUGE_BLOCKED<ROUGE_KERNEL_MR, ROUGE_KERNEL_NR, 1>(call);
    break;
  case 4:
    rouge_numerics_detail::ROUGE_BLOCKED<ROUGE_KERNEL_MR, ROUGE_KERNEL_NR, 4>(call);
    break;
  default:
    rouge_numerics_detail::ROUGE_BLOCKED<ROUGE_KERNEL_MR, ROUGE_KERNEL_NR, 2>(call);
    break;
  }
}

#undef ROUGE_PACK_PANEL
#undef ROUGE_MICRO
#undef ROUGE_EDGE
#undef ROUGE_PANEL
#undef ROUGE_BLOCKED
#undef ROUGE_PREFETCH
#undef ROUGE_RUN
#undef ROUGE_KERNEL_ELEM
#undef ROUGE_KERNEL_CALL
#undef ROUGE_KERNEL_PACK_A
#undef ROUGE_KERNEL_PACK_B