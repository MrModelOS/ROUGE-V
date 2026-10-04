/*
 * Автотюнер rouge-numerics: перебор кандидатов с честным замером wall-clock.
 *
 * Правила честности, которых этот файл придерживается:
 *   - замер идёт на настоящих данных A/B/C вызывающего, результат пишется в C;
 *   - кэша нет: каждый прогон действительно считает (warm-up прогоняется, но его
 *     время не учитывается);
 *   - из нескольких прогонов берётся минимум времени — так шум планировщика
 *     не завышает оценку кандидата;
 *   - в профиль попадает конфиг, который реально применялся, и GFLOP/s,
 *     посчитанные как 2*M*N*K / время.
 *
 * Никаких «ожидаемых» значений: отчёт строится только из измеренного.
 */

#include "rouge/numerics.h"

#include <algorithm>
#include <chrono>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

/* Кандидаты автотюнера. Смысл каждого набора — проверить разные режимы ядра:
 * мелкие плитки (меньше упаковки, меньше рабочего кэша), крупные плитки
 * (меньше проходов по A/B), разная глубина разворота цикла по K и разное
 * число частей K (split_k). Набор маленький и осмысленный: перебор тысяч
 * конфигов на 4 ядрах — это минуты на каждый вызов.
 */
const RougeGemmConfig kCandidates[] = {
    {32, 32, 32, 1, 1},   // мелкие плитки, без разворота
    {32, 64, 32, 2, 1},   // мелкие плитки, unroll 2
    {64, 64, 32, 2, 1},   // умолчание: умеренные плитки
    {64, 64, 64, 4, 1},   // глубокая k-панель и максимальный разворот
    {128, 128, 64, 4, 1}, // крупные плитки
    {32, 32, 32, 1, 2},   // деление K на 2 части
    {64, 64, 32, 2, 4},   // деление K на 4 части
};

constexpr int kCandidateCount = static_cast<int>(sizeof(kCandidates) / sizeof(kCandidates[0]));

/* GFLOP/s — это 10^9 FLOP/s. Без этого деления поле «gflops» на самом деле
 * содержало сырое число операций в секунду, завышая результат в миллиард раз
 * (именно так выглядели первые замеры: 1e11 «GFLOP/s» на четырёх ядрах). */
constexpr double kGiga = 1e9;

/* Сколько раз повторяем один замер. Одна итерация слишком шумна, поэтому
 * меряем несколько и берём лучшую. */
constexpr int kRepeats = 3;

/* Время одного прогона ядра в секундах. */
double time_once(int M, int N, int K, const float *A, const float *B, float *C,
                 const RougeGemmConfig &cfg) {
  const Clock::time_point t0 = Clock::now();
  rouge_gemm_f32(M, N, K, A, B, C, &cfg);
  const Clock::time_point t1 = Clock::now();
  return std::chrono::duration<double>(t1 - t0).count();
}

} // namespace

extern "C" {

int rouge_gemm_candidate_count(void) { return kCandidateCount; }

RougeGemmConfig rouge_gemm_candidate(int index) {
  if (index < 0 || index >= kCandidateCount)
    return rouge_gemm_default_config();
  return kCandidates[index];
}

double rouge_gemm_flops_per_sec(int M, int N, int K, const float *A, const float *B, float *C,
                                const RougeGemmConfig *cfg) {
  if (M <= 0 || N <= 0 || K <= 0 || !A || !B || !C)
    return 0.0;
  const RougeGemmConfig use = cfg ? *cfg : rouge_gemm_default_config();
  // Разогрев: первый прогон платит за холодные кэши и за аллокацию буферов,
  // его в оценку не берём — иначе измерялась бы не скорость ядра.
  time_once(M, N, K, A, B, C, use);
  double best = 0.0;
  for (int r = 0; r < kRepeats; ++r) {
    const double dt = time_once(M, N, K, A, B, C, use);
    if (best == 0.0 || dt < best)
      best = dt;
  }
  const double flops =
      2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
  return best > 0.0 ? flops / best / kGiga : 0.0;
}

RougeGemmProfile rouge_gemm_tuned(int M, int N, int K, const float *A, const float *B, float *C) {
  RougeGemmProfile prof;
  prof.cfg = rouge_gemm_default_config();
  prof.gflops = 0.0;
  prof.iters = 0;
  if (M <= 0 || N <= 0 || K <= 0 || !A || !B || !C)
    return prof;

  const double flops =
      2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);

  double best_dt = 0.0;
  long long iters = 0;
  for (int i = 0; i < kCandidateCount; ++i) {
    const RougeGemmConfig cfg = kCandidates[i];
    iters += 1; // разогрев кандидата — тоже считаем выполненным прогоном
    time_once(M, N, K, A, B, C, cfg);
    for (int r = 0; r < kRepeats; ++r) {
      iters += 1;
      const double dt = time_once(M, N, K, A, B, C, cfg);
      if (best_dt == 0.0 || dt < best_dt) {
        best_dt = dt;
        prof.cfg = cfg;
      }
    }
  }

  prof.iters = iters;
  prof.gflops = best_dt > 0.0 ? flops / best_dt / kGiga : 0.0;

  // C обязан содержать результат именно выбранного конфига: последний в цикле
  // мог быть другим кандидатом, поэтому считаем ещё раз.
  rouge_gemm_f32(M, N, K, A, B, C, &prof.cfg);
  return prof;
}

} // extern "C"