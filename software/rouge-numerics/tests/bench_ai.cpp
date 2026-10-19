// Замеры AI-части Wave 2: INT8-GEMM, GQA-внимание, трансформерный блок.
//
// Честность прежде всего:
//   * печатаем, ЧТО именно замерено (размеры, число прогонов, единицы);
//   * приводим GFLOP/s там, где операции считываемы;
//   * НИКАКИХ сравнений с чужими библиотеками: у нас нет ни cuBLAS, ни
//     FlashAttention, ни llama.cpp на том же железе и с тем же API;
//   * int8-путь пока СКАЛЯРНЫЙ (накопление в int32) — это точный, но
//     медленный эталон. Ускорение (SIMD/DotProd) — отдельная задача, и
//     показывать его как «оптимизировано» рано.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "rouge/numerics.h"

static double now_sec() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

template <typename F>
static double bench(int iters, F fn) {
  fn();  // разогрев
  double best = 1e300;
  for (int i = 0; i < iters; ++i) {
    const double t0 = now_sec();
    fn();
    const double t1 = now_sec();
    if (t1 - t0 < best) best = t1 - t0;
  }
  return best;
}

int main() {
  std::mt19937 rng(12345u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::printf("=== AI-часть rouge-numerics: замеры (лучшее из %d прогонов)\n",
              5);

  // ------------------------------------------------------ 1. INT8 GEMM ---
  {
    const int M = 256, N = 256, K = 256;
    std::vector<float> A((size_t)M * K), B((size_t)K * N);
    for (auto &v : A) v = dist(rng);
    for (auto &v : B) v = dist(rng);
    float sa = 0, sb = 0;
    std::vector<int8_t> qa(A.size()), qb(B.size());
    rouge_quantize_i8_tensor(A.data(), (int)A.size(), &sa, qa.data());
    rouge_quantize_i8_tensor(B.data(), (int)B.size(), &sb, qb.data());
    std::vector<float> C((size_t)M * N, 0.0f);

    const double t_i8 = bench(5, [&] {
      rouge_gemm_i8(M, N, K, qa.data(), qb.data(), &sa, 0, &sb, 0, C.data());
    });
    const double t_f32 = bench(5, [&] {
      rouge_gemm_f32(M, N, K, A.data(), B.data(), C.data(), nullptr);
    });
    const double ops = 2.0 * M * N * K;
    std::printf("GEMM %dx%dx%d:\n", M, N, K);
    std::printf("  int8 (скалярный, накопление int32): %7.2f мс, %6.2f GFLOP/s\n",
                t_i8 * 1e3, ops / t_i8 / 1e9);
    std::printf("  f32  (наш блочный, с автотюнером):    %7.2f мс, %6.2f GFLOP/s\n",
                t_f32 * 1e3, ops / t_f32 / 1e9);
    std::printf("  (int8 медленнее: SIMD-ядра пока нет — это точный эталон)\n");
  }

  // ------------------------------------------------ 2. GQA-внимание ---
  {
    const int S = 512, Dh = 64, Hq = 8, Hkv = 2;
    std::vector<float> Q((size_t)Hq * S * Dh), K((size_t)Hkv * S * Dh),
        V((size_t)Hkv * S * Dh), O((size_t)Hq * S * Dh);
    for (auto &v : Q) v = dist(rng);
    for (auto &v : K) v = dist(rng);
    for (auto &v : V) v = dist(rng);

    const double t_gqa = bench(3, [&] {
      rouge_attention_gqa(S, S, Dh, Hq, Hkv, Q.data(), K.data(), V.data(),
                          O.data(), 0.0f, 1, 0, 0);
    });
    // Одноголовое (MHA) — база для сравнения: у него в 4 раза больше K/V.
    const double t_mha = bench(3, [&] {
      rouge_attention_f32(S, S, Dh, Q.data(), K.data(), V.data(), O.data(), 0.0f,
                          1, 0);
    });
    // Оценка операций: 2*S*S*Dh на голову запроса (умножения QK и PV).
    const double ops_gqa = 2.0 * S * S * Dh * Hq;
    std::printf("внимание S=%d, Dh=%d, Hq=%d, Hkv=%d (causal):\n", S, Dh, Hq,
                Hkv);
    std::printf("  GQA: %7.2f мс, %6.2f GFLOP/s\n", t_gqa * 1e3,
                ops_gqa / t_gqa / 1e9);
    std::printf("  MHA (одна голова, тот же объём Q): %7.2f мс, %6.2f GFLOP/s\n",
                t_mha * 1e3, 2.0 * S * S * Dh / t_mha / 1e9);
    std::printf("  (GQA читает в %d раза меньше K/V — это и есть смысл GQA)\n",
                Hq / Hkv);
  }

  // --------------------------------------------- 3. Трансформерный блок ---
  {
    const int S = 256, D = 512, Hq = 8, Hkv = 2, Dq = 64, Dff = 1376;
    std::vector<float> x((size_t)S * D), wrms1(D), wrms2(D);
    std::vector<float> wq((size_t)Hq * Dq * D), wk((size_t)Hkv * Dq * D),
        wv((size_t)Hkv * Dq * D), wo((size_t)D * Hq * Dq);
    std::vector<float> wgate((size_t)Dff * D), wup((size_t)Dff * D),
        wdown((size_t)D * Dff);
    for (auto &v : x) v = dist(rng);
    for (auto &v : wrms1) v = 1.0f;
    for (auto &v : wrms2) v = 1.0f;
    for (auto &v : wq) v = dist(rng) * 0.05f;
    for (auto &v : wk) v = dist(rng) * 0.05f;
    for (auto &v : wv) v = dist(rng) * 0.05f;
    for (auto &v : wo) v = dist(rng) * 0.05f;
    for (auto &v : wgate) v = dist(rng) * 0.05f;
    for (auto &v : wup) v = dist(rng) * 0.05f;
    for (auto &v : wdown) v = dist(rng) * 0.05f;
    std::vector<float> out((size_t)S * D);

    const double t_blk = bench(3, [&] {
      rouge_transformer_block(S, D, Hq, Hkv, Dq, Dq, x.data(), wrms1.data(),
                               wq.data(), wk.data(), wv.data(), wo.data(),
                               wrms2.data(), wgate.data(), wup.data(),
                               wdown.data(), Dff, 10000.0f, 1e-5f, out.data());
    });
    // Оценка операций блока: проекции Q/K/V/O + MLP (умножения-accumulate).
    const double ops_proj = 2.0 * S * D * (Hq * Dq + 2 * Hkv * Dq + D);
    const double ops_mlp = 2.0 * S * D * 3 * Dff;
    const double ops_attn = 2.0 * S * S * Dq * Hq;
    const double ops = ops_proj + ops_mlp + ops_attn;
    std::printf("трансформерный блок S=%d, D=%d, Hq=%d, Hkv=%d, Dff=%d:\n", S, D,
                Hq, Hkv, Dff);
    std::printf("  %7.2f мс на блок, %6.2f GFLOP/s, %8.1f мкс/токен\n",
                t_blk * 1e3, ops / t_blk / 1e9, t_blk * 1e6 / S);
  }

  return 0;
}
