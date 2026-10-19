// Замер свёртки Wave 2 против oneDNN (открытый эталон).
//
// Одинаковые данные NCHW, одна и та же математика (cross-correlation, как в
// cuDNN и oneDNN): наш rouge_conv2d_f32 (im2col + свой блочный GEMM) против
// oneDNN convolution_forward. Корректность обязательна (допуск 1e-5 против
// oneDNN — обе стороны f32), скорость — только отчёт.
//
// Используется C++ API oneDNN (dnnl.hpp) — он же есть в пакете oneDNN.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include <dnnl.hpp>

#include "rouge/numerics.h"

static int failures = 0;

int main() {
  std::mt19937 rng(0x0D11u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  const dnnl::engine eng(dnnl::engine::kind::cpu, 0);
  const dnnl::stream strm(eng);

  const struct {
    int N, Cin, H, W, Cout, KH, KW, stride, pad;
  } cases[] = {
      {1, 16, 32, 32, 16, 3, 3, 1, 1},
      {2, 32, 32, 32, 32, 3, 3, 1, 1},
      {1, 16, 28, 28, 32, 3, 3, 1, 1},
      {2, 8, 16, 16, 16, 5, 5, 2, 2},
      // Крупные формы: тут oneDNN может выиграть (у него brgemm с JIT) —
      // сравнение должно быть честным в обе стороны.
      {1, 64, 56, 56, 64, 3, 3, 1, 1},
      {4, 32, 32, 32, 64, 3, 3, 1, 1},
      {1, 3, 224, 224, 64, 7, 7, 2, 3},
  };

  for (const auto &cs : cases) {
    const int OH = (cs.H + 2 * cs.pad - cs.KH) / cs.stride + 1;
    const int OW = (cs.W + 2 * cs.pad - cs.KW) / cs.stride + 1;
    std::vector<float> X((size_t)cs.N * cs.Cin * cs.H * cs.W);
    std::vector<float> Wt((size_t)cs.Cout * cs.Cin * cs.KH * cs.KW);
    for (auto &v : X) v = dist(rng);
    for (auto &v : Wt) v = dist(rng);

    const dnnl::memory::desc x_md({cs.N, cs.Cin, cs.H, cs.W},
                                   dnnl::memory::data_type::f32,
                                   dnnl::memory::format_tag::nchw);
    const dnnl::memory::desc w_md({cs.Cout, cs.Cin, cs.KH, cs.KW},
                                   dnnl::memory::data_type::f32,
                                   dnnl::memory::format_tag::oihw);
    const dnnl::memory::desc y_md({cs.N, cs.Cout, OH, OW},
                                   dnnl::memory::data_type::f32,
                                   dnnl::memory::format_tag::nchw);

    dnnl::memory x(x_md, eng, X.data());
    dnnl::memory w(w_md, eng, Wt.data());
    std::vector<float> Ydnnl((size_t)cs.N * cs.Cout * OH * OW, 0.0f);
    dnnl::memory y(y_md, eng, Ydnnl.data());

    const dnnl::memory::dims strides = {cs.stride, cs.stride};
    const dnnl::memory::dims pad_l = {cs.pad, cs.pad};
    const dnnl::memory::dims pad_r = {cs.pad, cs.pad};
    // algorithm::convolution_auto — oneDNN сам выбирает лучшую реализацию
    // (direct/winograd/brgemm): иначе сравнение было бы нечестным в его пользу
    // или против него, в зависимости от того, что выбрано по умолчанию.
    dnnl::convolution_forward::primitive_desc conv_pd(
        eng, dnnl::prop_kind::forward_inference,
        dnnl::algorithm::convolution_auto, x_md, w_md, dnnl::memory::desc(),
        y_md, strides, pad_l, pad_r, dnnl::primitive_attr());
    dnnl::primitive conv(conv_pd);

    std::vector<float> Yrouge((size_t)cs.N * cs.Cout * OH * OW, 0.0f);
    auto t_ours = [&] {
      rouge_conv2d_f32(cs.N, cs.Cin, cs.H, cs.W, cs.Cout, cs.KH, cs.KW,
                       cs.stride, cs.pad, X.data(), Wt.data(), nullptr,
                       Yrouge.data(), nullptr);
    };
    auto t_dnnl = [&] {
      conv.execute(strm, {{DNNL_ARG_SRC, x}, {DNNL_ARG_WEIGHTS, w}, {DNNL_ARG_DST, y}});
    };

    t_ours();
    t_dnnl();
    double md = 0.0, sc = 1.0;
    for (size_t i = 0; i < Ydnnl.size(); ++i) {
      md = std::max(md, std::fabs((double)Yrouge[i] - (double)Ydnnl[i]));
      sc = std::max(sc, std::fabs((double)Ydnnl[i]));
    }
    if (md / sc > 1e-5) {
      std::printf("FAIL conv %dx%d %dx%d s=%d p=%d vs oneDNN: rel %g\n",
                  cs.Cin, cs.Cout, cs.KH, cs.KW, cs.stride, cs.pad, md / sc);
      ++failures;
    }

    auto bench = [&](auto &&fn) {
      fn();
      double best = 1e300;
      for (int i = 0; i < 5; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        fn();
        auto t1 = std::chrono::steady_clock::now();
        best = std::min(best,
                        std::chrono::duration<double, std::milli>(t1 - t0).count());
      }
      return best;
    };
    const double t_r = bench(t_ours);
    const double t_d = bench(t_dnnl);
    std::printf(
        "conv N=%d %d->%d %dx%d k%dx%d s%d p%d: rouge %7.3f ms  oneDNN %7.3f ms  (rel %g)\n",
        cs.N, cs.Cin, cs.Cout, cs.H, cs.W, cs.KH, cs.KW, cs.stride, cs.pad, t_r,
        t_d, md / sc);
  }
  if (failures == 0) std::printf("PASS: conv vs oneDNN\n");
  return failures ? 1 : 0;
}
