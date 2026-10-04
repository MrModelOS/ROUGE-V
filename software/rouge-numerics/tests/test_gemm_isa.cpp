// Тест rouge-numerics: один и тот же набор конфигов против наивного эталона, но
// для КАЖДОГО собранного варианта ISA (baseline / avx2+fma / avx512f+fma).
//
// Зачем: на одной машине процессор умеет не всё, поэтому проверяется только
// один путь. Вариант выбирается переменной окружения ROUGE_NUMERICS_ISA,
// которую ctest ставит для каждого варианта отдельно. Если процессор не умеет
// запрошенный вариант, библиотека его отклоняет — тест это увидит по
// rouge_gemm_isa_variant() и честно скажет, что путь не проверялся.

#include <rouge/numerics.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

struct Size {
  int m, n, k;
};

// Те же размеры, что и в test_gemm.cpp: некратные плитке плюс крупные.
const Size kSizes[] = {
    {1, 1, 1}, {3, 5, 7}, {17, 33, 65}, {128, 128, 128}, {200, 100, 50}, {129, 131, 127},
};

constexpr double kTol = 1e-5;

// max|C - ref| / max(||ref||_inf, 1); единица в знаменателе защищает от деления
// на почти-нулевую сумму при K=1, не маскируя ошибку (элементы порядка единицы).
double rel_error(const std::vector<float> &got, const std::vector<double> &ref) {
  double maxdiff = 0.0;
  double scale = 1.0;
  for (std::size_t t = 0; t < ref.size(); ++t) {
    maxdiff = std::max(maxdiff, std::fabs(static_cast<double>(got[t]) - ref[t]));
    scale = std::max(scale, std::fabs(ref[t]));
  }
  return maxdiff / scale;
}

// Наивная тройная петля в double, чтобы погрешность эталона не влияла на оценку.
std::vector<double> naive_ref(int M, int N, int K, const std::vector<float> &A,
                              const std::vector<float> &B) {
  std::vector<double> ref(static_cast<std::size_t>(M) * N, 0.0);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      double s = 0.0;
      for (int k = 0; k < K; ++k) {
        s += static_cast<double>(A[static_cast<std::size_t>(i) * K + k]) *
             static_cast<double>(B[static_cast<std::size_t>(k) * N + j]);
      }
      ref[static_cast<std::size_t>(i) * N + j] = s;
    }
  }
  return ref;
}

} // namespace

int main() {
  const char *want = std::getenv("ROUGE_NUMERICS_ISA");
  const std::string want_name = want ? want : "";
  const char *got = rouge_gemm_isa_variant();
  const std::string got_name = got ? got : "";

  std::printf("запрошенный вариант ISA: %s, библиотека считает: %s\n", want_name.c_str(),
              got_name.c_str());

  // Сверяем по имени без префикса "+fma": переменная окружения задаёт семейство.
  auto family = [](const std::string &s) { return s.substr(0, s.find('+')); };
  if (!want_name.empty() && family(got_name) != family(want_name)) {
    std::printf("SKIP: процессор не поддерживает вариант %s (проверять нечего)\n",
                want_name.c_str());
    return 0;
  }

  const int n_cand = rouge_gemm_candidate_count();
  double worst_rel = 0.0;
  int failures = 0;

  for (const Size &sz : kSizes) {
    const int M = sz.m, N = sz.n, K = sz.k;
    std::mt19937 rng(static_cast<unsigned>(M * 7919 + N * 104729 + K));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> A(static_cast<std::size_t>(M) * K);
    std::vector<float> B(static_cast<std::size_t>(K) * N);
    for (float &x : A)
      x = dist(rng);
    for (float &x : B)
      x = dist(rng);
    const std::vector<double> ref = naive_ref(M, N, K, A, B);

    for (int ci = 0; ci < n_cand; ++ci) {
      const RougeGemmConfig cfg = rouge_gemm_candidate(ci);
      std::vector<float> C(static_cast<std::size_t>(M) * N, -7.0f);
      rouge_gemm_f32(M, N, K, A.data(), B.data(), C.data(), &cfg);
      const double rel = rel_error(C, ref);
      worst_rel = std::max(worst_rel, rel);
      if (!(std::isfinite(rel) && rel <= kTol))
        ++failures;
    }

    // Автотюнер на том же варианте ISA: профиль обязан быть непустым, а C —
    // совпасть с эталоном.
    std::vector<float> C(static_cast<std::size_t>(M) * N, -7.0f);
    const RougeGemmProfile prof = rouge_gemm_tuned(M, N, K, A.data(), B.data(), C.data());
    const double rel = rel_error(C, ref);
    worst_rel = std::max(worst_rel, rel);
    if (!(std::isfinite(rel) && rel <= kTol && prof.gflops > 0.0 && prof.iters > 0))
      ++failures;

    std::printf("  M=%-3d N=%-3d K=%-3d  %d конфигов + автотюнер (tm=%d tn=%d tk=%d unroll=%d "
                "split_k=%d): худшая rel_err=%.3e\n",
                M, N, K, n_cand, prof.cfg.tile_m, prof.cfg.tile_n, prof.cfg.tile_k, prof.cfg.unroll,
                prof.cfg.split_k, worst_rel);
  }

  std::printf("ISA %s: худшая относительная погрешность = %.3e (допуск %.0e), расхождений: %d\n",
              got_name.c_str(), worst_rel, kTol, failures);
  if (failures != 0) {
    std::printf("FAIL: вариант ISA %s не сошёлся с наивным эталоном\n", got_name.c_str());
    return 1;
  }
  std::printf("PASS: вариант ISA %s совпал с наивным эталоном по всем конфигам\n",
              got_name.c_str());
  return 0;
}