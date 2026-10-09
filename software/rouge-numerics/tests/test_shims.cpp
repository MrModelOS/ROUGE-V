// Шимы cuBLAS/cuDNN: счёт через наш код против независимых эталонов.
//
// cublasSgemm_v2: все 4 комбинации N/T на случайных матрицах против
//   эталона в double (column-major, с alpha/beta). Плюс Create/Destroy,
//   версия, статусы, отказ на мусорных lda.
// cudnnConvolutionForward: NCHW f32 против rouge_conv2d_f32, оба алгоритма
//   (IMPLICIT_GEMM и WINOGRAD), alpha/beta-масштабирование, dims,
//   отказ на HALF и на dilation.

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "rouge/cublas_shim.h"
#include "rouge/cudnn_shim.h"
#include "rouge/numerics.h"

static int failures = 0;

#define EXPECT_TRUE(cond)                                        \
  do {                                                           \
    if (!(cond)) {                                               \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                \
    }                                                            \
  } while (0)

// Эталон SGEMM в double, column-major.
static void ref_sgemm(cublasOperation_t ta, cublasOperation_t tb, int m, int n,
                      int k, float al, const std::vector<float> &A, int lda,
                      const std::vector<float> &B, int ldb, float be,
                      std::vector<float> &C, int ldc) {
  auto opA = [&](int i, int j) -> double {
    return (ta == CUBLAS_OP_N) ? A[(size_t)j * lda + i] : A[(size_t)i * lda + j];
  };
  auto opB = [&](int i, int j) -> double {
    return (tb == CUBLAS_OP_N) ? B[(size_t)j * ldb + i] : B[(size_t)i * ldb + j];
  };
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) {
      double s = 0;
      for (int p = 0; p < k; ++p) s += opA(i, p) * opB(p, j);
      C[(size_t)j * ldc + i] = (float)(al * s + be * C[(size_t)j * ldc + i]);
    }
}

int main() {
  std::mt19937 rng(0x5A47u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  // --- cuBLAS ---
  cublasHandle_t h = nullptr;
  EXPECT_TRUE(cublasCreate_v2(&h) == CUBLAS_STATUS_SUCCESS && h);
  EXPECT_TRUE(cublasCreate_v2(nullptr) == CUBLAS_STATUS_INVALID_VALUE);
  int ver = 0;
  EXPECT_TRUE(cublasGetVersion_v2(h, &ver) == CUBLAS_STATUS_SUCCESS);
  EXPECT_TRUE(ver == 300);
  EXPECT_TRUE(cublasGetStatusString(CUBLAS_STATUS_SUCCESS) != nullptr);

  const int m = 5, n = 4, k = 6;
  for (int combo = 0; combo < 4; ++combo) {
    const auto ta =
        (combo & 1) ? CUBLAS_OP_T : CUBLAS_OP_N;
    const auto tb =
        (combo & 2) ? CUBLAS_OP_T : CUBLAS_OP_N;
    const int lda = (ta == CUBLAS_OP_N ? m : k) + 1;  // с запасом
    const int ldb = (tb == CUBLAS_OP_N ? k : n) + 1;
    const int ldc = m + 1;
    std::vector<float> A((size_t)lda * (ta == CUBLAS_OP_N ? k : m));
    std::vector<float> B((size_t)ldb * (tb == CUBLAS_OP_N ? n : k));
    std::vector<float> C((size_t)ldc * n), ref = C;
    for (auto &v : A) v = dist(rng);
    for (auto &v : B) v = dist(rng);
    for (auto &v : C) v = dist(rng);
    ref = C;
    const float al = 0.5f, be = 1.5f;
    EXPECT_TRUE(cublasSgemm_v2(h, ta, tb, m, n, k, &al, A.data(), lda,
                               B.data(), ldb, &be, C.data(), ldc) ==
                CUBLAS_STATUS_SUCCESS);
    ref_sgemm(ta, tb, m, n, k, al, A, lda, B, ldb, be, ref, ldc);
    double md = 0, sc = 1;
    for (size_t i = 0; i < ref.size(); ++i) {
      // Сравниваем только рабочую m×n область (хвосты lda/ldc — мусор).
      md = std::max(md, std::fabs((double)C[i] - ref[i]));
      sc = std::max(sc, std::fabs((double)ref[i]));
    }
    if (md / sc > 1e-5) {
      std::printf("sgemm combo %d rel %g\n", combo, md / sc);
      ++failures;
    }
  }
  {
    float a = 1, b = 0;
    std::vector<float> A(4, 1), B(4, 1), C(4, 0);
    EXPECT_TRUE(cublasSgemm_v2(h, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, 2, &a,
                               A.data(), 1, B.data(), 2, &b, C.data(), 2) ==
                CUBLAS_STATUS_INVALID_VALUE);  // lda < m
    EXPECT_TRUE(cublasSgemm_v2(nullptr, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, 2, &a,
                               A.data(), 2, B.data(), 2, &b, C.data(), 2) ==
                CUBLAS_STATUS_NOT_INITIALIZED);
  }
  EXPECT_TRUE(cublasDestroy_v2(h) == CUBLAS_STATUS_SUCCESS);

  // --- Dgemm: одна комбинация N/N + T/T против double-эталона ---
  EXPECT_TRUE(cublasCreate_v2(&h) == CUBLAS_STATUS_SUCCESS);
  {
    const int m = 6, n = 5, k = 7;
    std::vector<double> A(m * k), B(k * n), C(m * n), ref(m * n);
    for (auto &v : A) v = dist(rng);
    for (auto &v : B) v = dist(rng);
    for (auto &v : C) v = dist(rng);
    for (size_t i = 0; i < C.size(); ++i) ref[i] = C[i];  // как ref = C, но
    // без ложного -Warray-bounds GCC 16 на vector::operator= при -O2
    const double al = -0.5, be = 2.0;
    EXPECT_TRUE(cublasDgemm_v2(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &al,
                               A.data(), m, B.data(), k, &be, C.data(), m) ==
                CUBLAS_STATUS_SUCCESS);
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < m; ++i) {
        long double s = 0;
        for (int p = 0; p < k; ++p) s += (long double)A[p * m + i] * B[j * k + p];
        ref[j * m + i] = al * (double)s + be * ref[j * m + i];
      }
    double md = 0, sc = 1;
    for (size_t i = 0; i < ref.size(); ++i) {
      md = std::max(md, std::fabs(C[i] - ref[i]));
      sc = std::max(sc, std::fabs(ref[i]));
    }
    EXPECT_TRUE(md / sc < 1e-14);
    // T/T с теми же данными, переложенными вызывающим.
    std::vector<double> At(k * m), Bt(n * k);
    for (int i = 0; i < m; ++i)
      for (int p = 0; p < k; ++p) At[i * k + p] = A[p * m + i];
    for (int p = 0; p < k; ++p)
      for (int j = 0; j < n; ++j) Bt[p * n + j] = B[j * k + p];
    std::fill(C.begin(), C.end(), 0.0);
    EXPECT_TRUE(cublasDgemm_v2(h, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, &al,
                               At.data(), k, Bt.data(), n, &be, C.data(), m) ==
                CUBLAS_STATUS_SUCCESS);
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < m; ++i) {
        long double s = 0;
        for (int p = 0; p < k; ++p) s += (long double)A[p * m + i] * B[j * k + p];
        if (std::fabs(C[j * m + i] - (al * (double)s)) > 1e-12 * (1 + std::fabs(C[j * m + i])))
          ++failures;
      }
  }

  // --- Cgemm/Zgemm против long double-эталона (включая сопряжение C) ---
  {
    const int m = 4, n = 3, k = 5;
    std::vector<cuComplex> A(m * k), C(m * n);
    // Для op(B)=C: B лежит n×k с ldb>=n (op(B) — k×n).
    const int ldb = n + 1;
    std::vector<cuComplex> B((size_t)ldb * k);
    for (auto &v : A) {
      v.x = dist(rng);
      v.y = dist(rng);
    }
    for (auto &v : B) {
      v.x = dist(rng);
      v.y = dist(rng);
    }
    for (auto &v : C) {
      v.x = dist(rng);
      v.y = dist(rng);
    }
    const cuComplex al{0.5f, -0.25f}, be{1.0f, 0.5f};
    std::vector<cuComplex> ref = C;
    EXPECT_TRUE(cublasCgemm_v2(h, CUBLAS_OP_N, CUBLAS_OP_C, m, n, k, &al,
                               A.data(), m, B.data(), ldb, &be, C.data(),
                               m) == CUBLAS_STATUS_SUCCESS);
    // Эталон: op(B)[p][j] = conj(B_stored[j][p]), B_stored — n×k col-major.
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < m; ++i) {
        long double sr = 0, si = 0;
        for (int p = 0; p < k; ++p) {
          const long double ar = A[p * m + i].x, ai = A[p * m + i].y;
          const long double br = B[p * ldb + j].x, bi = -B[p * ldb + j].y;
          sr += ar * br - ai * bi;
          si += ar * bi + ai * br;
        }
        const long double wr =
            (al.x * sr - al.y * si) + (be.x * ref[j * m + i].x - be.y * ref[j * m + i].y);
        const long double wi =
            (al.x * si + al.y * sr) + (be.x * ref[j * m + i].y + be.y * ref[j * m + i].x);
        if (std::fabs((double)C[j * m + i].x - (double)wr) > 1e-5 * (1 + std::fabs((double)wr)) ||
            std::fabs((double)C[j * m + i].y - (double)wi) > 1e-5 * (1 + std::fabs((double)wi))) {
          std::printf("cgemm N/C mismatch @%d,%d\n", i, j);
          ++failures;
        }
      }
    // Zgemm N/N.
    std::vector<cuDoubleComplex> Ad(m * k), Bd(k * n), Cd(m * n);
    for (auto &v : Ad) {
      v.x = dist(rng);
      v.y = dist(rng);
    }
    for (auto &v : Bd) {
      v.x = dist(rng);
      v.y = dist(rng);
    }
    for (auto &v : Cd) {
      v.x = dist(rng);
      v.y = dist(rng);
    }
    const cuDoubleComplex ald{2.0, 0.5}, bed{0.0, 1.0};
    std::vector<cuDoubleComplex> refd = Cd;
    EXPECT_TRUE(cublasZgemm_v2(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &ald,
                               Ad.data(), m, Bd.data(), k, &bed, Cd.data(),
                               m) == CUBLAS_STATUS_SUCCESS);
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < m; ++i) {
        long double sr = 0, si = 0;
        for (int p = 0; p < k; ++p) {
          const long double ar = Ad[p * m + i].x, ai = Ad[p * m + i].y;
          const long double br = Bd[j * k + p].x, bi = Bd[j * k + p].y;
          sr += ar * br - ai * bi;
          si += ar * bi + ai * br;
        }
        const long double wr = (ald.x * sr - ald.y * si) +
                               (bed.x * refd[j * m + i].x - bed.y * refd[j * m + i].y);
        const long double wi = (ald.x * si + ald.y * sr) +
                               (bed.x * refd[j * m + i].y + bed.y * refd[j * m + i].x);
        if (std::fabs(Cd[j * m + i].x - (double)wr) > 1e-12 * (1 + std::fabs((double)wr)) ||
            std::fabs(Cd[j * m + i].y - (double)wi) > 1e-12 * (1 + std::fabs((double)wi))) {
          std::printf("zgemm mismatch @%d,%d\n", i, j);
          ++failures;
        }
      }
  }

  // --- Strided-batched S/D: 3 пачки против поштучного эталона ---
  {
    const int m = 4, n = 4, k = 4, batch = 3;
    std::vector<float> A(batch * m * k), B(batch * k * n), C(batch * m * n),
        ref(batch * m * n);
    for (auto &v : A) v = dist(rng);
    for (auto &v : B) v = dist(rng);
    const float al = 1.0f, be = 0.0f;
    EXPECT_TRUE(cublasSgemmStridedBatched(
                    h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &al, A.data(), m,
                    m * k, B.data(), k, k * n, &be, C.data(), m, m * n,
                    batch) == CUBLAS_STATUS_SUCCESS);
    for (int b = 0; b < batch; ++b)
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
          double s = 0;
          for (int p = 0; p < k; ++p)
            s += (double)A[b * m * k + p * m + i] * B[b * k * n + j * k + p];
          ref[b * m * n + j * m + i] = (float)s;
        }
    double md = 0, sc = 1;
    for (size_t i = 0; i < ref.size(); ++i) {
      md = std::max(md, std::fabs((double)C[i] - ref[i]));
      sc = std::max(sc, std::fabs((double)ref[i]));
    }
    EXPECT_TRUE(md / sc < 1e-5);
    EXPECT_TRUE(cublasSgemmStridedBatched(
                    h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &al, A.data(), m,
                    m * k, B.data(), k, k * n, &be, C.data(), m, m * n,
                    -1) == CUBLAS_STATUS_INVALID_VALUE);
    // D-batched smoke (полное сравнение — через тот же приём, короче).
    std::vector<double> Ad2(batch * m * k, 0.5), Bd2(batch * k * n, 0.25),
        Cd2(batch * m * n, 0.0);
    const double ald = 1.0, bed = 0.0;
    EXPECT_TRUE(cublasDgemmStridedBatched(
                    h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &ald, Ad2.data(), m,
                    m * k, Bd2.data(), k, k * n, &bed, Cd2.data(), m, m * n,
                    batch) == CUBLAS_STATUS_SUCCESS);
    for (double v : Cd2) EXPECT_TRUE(std::fabs(v - 0.5) < 1e-12);
  }
  EXPECT_TRUE(cublasDestroy_v2(h) == CUBLAS_STATUS_SUCCESS);

  // --- cuDNN ---
  cudnnHandle_t dh = nullptr;
  EXPECT_TRUE(cudnnCreate(&dh) == CUDNN_STATUS_SUCCESS && dh);
  cudnnTensorDescriptor_t xD = nullptr, yD = nullptr;
  cudnnFilterDescriptor_t wD = nullptr;
  cudnnConvolutionDescriptor_t conv = nullptr;
  cudnnCreateTensorDescriptor(&xD);
  cudnnCreateTensorDescriptor(&yD);
  cudnnCreateFilterDescriptor(&wD);
  cudnnCreateConvolutionDescriptor(&conv);
  EXPECT_TRUE(cudnnSetTensor4dDescriptor(xD, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                         1, 2, 8, 8) == CUDNN_STATUS_SUCCESS);
  EXPECT_TRUE(cudnnSetFilter4dDescriptor(wD, CUDNN_DATA_FLOAT,
                                         CUDNN_TENSOR_NCHW, 3, 2, 3, 3) ==
              CUDNN_STATUS_SUCCESS);
  EXPECT_TRUE(cudnnSetConvolution2dDescriptor(conv, 1, 1, 1, 1, 1, 1, 0,
                                              CUDNN_DATA_FLOAT) ==
              CUDNN_STATUS_SUCCESS);
  int on, oc, oh, ow;
  EXPECT_TRUE(cudnnGetConvolution2dForwardOutputDim(conv, xD, wD, &on, &oc, &oh,
                                                    &ow) == CUDNN_STATUS_SUCCESS);
  EXPECT_TRUE(on == 1 && oc == 3 && oh == 8 && ow == 8);
  EXPECT_TRUE(cudnnSetTensor4dDescriptor(yD, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                         on, oc, oh, ow) == CUDNN_STATUS_SUCCESS);
  std::vector<float> X(1 * 2 * 8 * 8), W(3 * 2 * 9), Y(1 * 3 * 8 * 8, 0.25f),
      ref(Y.size(), 0);
  for (auto &v : X) v = dist(rng);
  for (auto &v : W) v = dist(rng);
  ref = Y;
  const float al = 2.0f, be = 0.5f;
  for (int algo = 0; algo < 2; ++algo) {
    std::vector<float> got = Y;
    const auto alg = algo == 0 ? CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
                               : CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
    EXPECT_TRUE(cudnnConvolutionForward(dh, &al, xD, X.data(), wD, W.data(),
                                        conv, alg, nullptr, 0, &be, yD,
                                        got.data()) == CUDNN_STATUS_SUCCESS);
    std::vector<float> base(ref.size(), 0);
    rouge_conv2d_f32(1, 2, 8, 8, 3, 3, 3, 1, 1, X.data(), W.data(), nullptr,
                     base.data(), nullptr);
    double md = 0, sc = 1;
    for (size_t i = 0; i < ref.size(); ++i) {
      const double want = al * base[i] + be * ref[i];
      md = std::max(md, std::fabs((double)got[i] - want));
      sc = std::max(sc, std::fabs(want));
    }
    if (md / sc > 1e-4) {
      std::printf("cudnn algo %d rel %g\n", algo, md / sc);
      ++failures;
    }
  }
  // Отказы: HALF-тензор и dilation 2.
  cudnnTensorDescriptor_t halfD = nullptr;
  cudnnCreateTensorDescriptor(&halfD);
  cudnnSetTensor4dDescriptor(halfD, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, 1, 2, 8,
                             8);
  EXPECT_TRUE(cudnnConvolutionForward(dh, &al, halfD, X.data(), wD, W.data(),
                                      conv,
                                      CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                                      nullptr, 0, &be, yD, Y.data()) ==
              CUDNN_STATUS_NOT_SUPPORTED);
  EXPECT_TRUE(cudnnSetConvolution2dDescriptor(conv, 1, 1, 1, 1, 2, 2, 0,
                                              CUDNN_DATA_FLOAT) ==
              CUDNN_STATUS_NOT_SUPPORTED);
  cudnnDestroyTensorDescriptor(xD);
  cudnnDestroyTensorDescriptor(yD);
  cudnnDestroyTensorDescriptor(halfD);
  cudnnDestroyFilterDescriptor(wD);
  cudnnDestroyConvolutionDescriptor(conv);

  // --- cudnnAddTensor / Activation / Softmax / BatchNorm ---
  cudnnTensorDescriptor_t tD = nullptr, tD2 = nullptr;
  cudnnCreateTensorDescriptor(&tD);
  cudnnCreateTensorDescriptor(&tD2);
  cudnnSetTensor4dDescriptor(tD, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 2, 3, 4,
                             5);
  cudnnSetTensor4dDescriptor(tD2, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 2, 3, 4,
                             5);
  std::vector<float> T1(2 * 3 * 4 * 5), T2(T1.size());
  for (auto &v : T1) v = dist(rng);
  for (auto &v : T2) v = dist(rng);
  {
    const float a1 = 2.0f, a2 = -1.0f;
    std::vector<float> ref = T2;
    EXPECT_TRUE(cudnnAddTensor(dh, &a1, tD, T1.data(), &a2, tD2, T2.data()) ==
                CUDNN_STATUS_SUCCESS);
    for (size_t i = 0; i < ref.size(); ++i)
      EXPECT_TRUE(T2[i] == a1 * T1[i] + a2 * ref[i]);
    // Несовпадающие dims — громкий отказ.
    cudnnTensorDescriptor_t bad = nullptr;
    cudnnCreateTensorDescriptor(&bad);
    cudnnSetTensor4dDescriptor(bad, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 2, 3,
                               4, 4);
    EXPECT_TRUE(cudnnAddTensor(dh, &a1, tD, T1.data(), &a2, bad, T2.data()) ==
                CUDNN_STATUS_NOT_SUPPORTED);
    cudnnDestroyTensorDescriptor(bad);
  }
  {
    cudnnActivationDescriptor_t act = nullptr;
    cudnnCreateActivationDescriptor(&act);
    const float al = 1.0f, be = 0.0f;
    // RELU.
    EXPECT_TRUE(cudnnSetActivationDescriptor(act, CUDNN_ACTIVATION_RELU,
                                             CUDNN_NOT_PROPAGATE_NAN, 0.0) ==
                CUDNN_STATUS_SUCCESS);
    T2 = T1;
    EXPECT_TRUE(cudnnActivationForward(dh, act, &al, tD, T1.data(), &be, tD2,
                                       T2.data()) == CUDNN_STATUS_SUCCESS);
    for (size_t i = 0; i < T1.size(); ++i)
      EXPECT_TRUE(T2[i] == (T1[i] > 0 ? T1[i] : 0.0f));
    // SIGMOID против double-эталона.
    EXPECT_TRUE(cudnnSetActivationDescriptor(act, CUDNN_ACTIVATION_SIGMOID,
                                             CUDNN_NOT_PROPAGATE_NAN, 0.0) ==
                CUDNN_STATUS_SUCCESS);
    EXPECT_TRUE(cudnnActivationForward(dh, act, &al, tD, T1.data(), &be, tD2,
                                       T2.data()) == CUDNN_STATUS_SUCCESS);
    for (size_t i = 0; i < T1.size(); ++i) {
      const double want = 1.0 / (1.0 + std::exp(-(double)T1[i]));
      if (std::fabs((double)T2[i] - want) > 1e-6 * (1 + std::fabs(want)))
        ++failures;
    }
    // TANH + CLIPPED_RELU smoke.
    EXPECT_TRUE(cudnnSetActivationDescriptor(act, CUDNN_ACTIVATION_TANH,
                                             CUDNN_NOT_PROPAGATE_NAN, 0.0) ==
                CUDNN_STATUS_SUCCESS);
    EXPECT_TRUE(cudnnActivationForward(dh, act, &al, tD, T1.data(), &be, tD2,
                                       T2.data()) == CUDNN_STATUS_SUCCESS);
    EXPECT_TRUE(cudnnSetActivationDescriptor(act, CUDNN_ACTIVATION_CLIPPED_RELU,
                                             CUDNN_NOT_PROPAGATE_NAN, 6.0) ==
                CUDNN_STATUS_SUCCESS);
    EXPECT_TRUE(cudnnActivationForward(dh, act, &al, tD, T1.data(), &be, tD2,
                                       T2.data()) == CUDNN_STATUS_SUCCESS);
    for (size_t i = 0; i < T1.size(); ++i)
      EXPECT_TRUE(T2[i] >= 0.0f && T2[i] <= 6.0f);
    cudnnDestroyActivationDescriptor(act);
  }
  {
    // Softmax INSTANCE: суммы по строкам == 1; CHANNEL: суммы по C == 1.
    const float al = 1.0f, be = 0.0f;
    T2.assign(T2.size(), 0.0f);
    EXPECT_TRUE(cudnnSoftmaxForward(dh, CUDNN_SOFTMAX_ACCURATE,
                                    CUDNN_SOFTMAX_MODE_INSTANCE, &al, tD,
                                    T1.data(), &be, tD2, T2.data()) ==
                CUDNN_STATUS_SUCCESS);
    for (int e = 0; e < 2; ++e) {
      double s = 0;
      for (int i = 0; i < 3 * 4 * 5; ++i) {
        EXPECT_TRUE(T2[e * 60 + i] >= 0.0f);
        s += T2[e * 60 + i];
      }
      EXPECT_TRUE(std::fabs(s - 1.0) < 1e-5);
    }
    EXPECT_TRUE(cudnnSoftmaxForward(dh, CUDNN_SOFTMAX_LOG,
                                    CUDNN_SOFTMAX_MODE_CHANNEL, &al, tD,
                                    T1.data(), &be, tD2, T2.data()) ==
                CUDNN_STATUS_SUCCESS);
    for (int e = 0; e < 2; ++e)
      for (int hw = 0; hw < 20; ++hw) {
        double s = 0;
        for (int c = 0; c < 3; ++c) s += std::exp(T2[(e * 3 + c) * 20 + hw]);
        if (std::fabs(s - 1.0) > 1e-4) ++failures;
      }
  }
  {
    // BatchNorm inference: y = (x-mean)/sqrt(var+eps)*scale + bias.
    cudnnTensorDescriptor_t pD = nullptr;
    cudnnCreateTensorDescriptor(&pD);
    cudnnSetTensor4dDescriptor(pD, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 3,
                               1, 1);
    std::vector<float> scale = {2.0f, 0.5f, 1.0f}, biasv = {1.0f, 0.0f, -1.0f};
    std::vector<float> mean = {0.5f, 0.0f, 1.0f}, var = {0.25f, 1.0f, 4.0f};
    const float al = 1.0f, be = 0.0f;
    T2.assign(T2.size(), 0.0f);
    EXPECT_TRUE(cudnnBatchNormalizationForwardInference(
                    dh, CUDNN_BATCHNORM_SPATIAL, &al, &be, tD, T1.data(), tD2,
                    T2.data(), pD, scale.data(), biasv.data(), mean.data(),
                    var.data(), 1e-5) == CUDNN_STATUS_SUCCESS);
    for (int c = 0; c < 3; ++c) {
      const double inv = 1.0 / std::sqrt(var[c] + 1e-5);
      for (int e = 0; e < 2; ++e)
        for (int hw = 0; hw < 20; ++hw) {
          const size_t o = ((size_t)e * 3 + c) * 20 + hw;
          const double want =
              (T1[o] - mean[c]) * inv * scale[c] + biasv[c];
          if (std::fabs((double)T2[o] - want) > 1e-5 * (1 + std::fabs(want)))
            ++failures;
        }
    }
    // Плохой epsilon — отказ.
    EXPECT_TRUE(cudnnBatchNormalizationForwardInference(
                    dh, CUDNN_BATCHNORM_SPATIAL, &al, &be, tD, T1.data(), tD2,
                    T2.data(), pD, scale.data(), biasv.data(), mean.data(),
                    var.data(), -1.0) == CUDNN_STATUS_BAD_PARAM);
    cudnnDestroyTensorDescriptor(pD);
  }
  cudnnDestroyTensorDescriptor(tD);
  cudnnDestroyTensorDescriptor(tD2);

  // --- cudnn conv backward: workspace 0 + dx/dw против прямых формул ---
  {
    // Маленькая конфигурация 1x1x4x4, ядро 3x3 s1p0 -> OH=OW=2.
    cudnnTensorDescriptor_t xDb = nullptr, dyDb = nullptr, dxDb = nullptr;
    cudnnFilterDescriptor_t wDb = nullptr, dwDb = nullptr;
    cudnnConvolutionDescriptor_t convDb = nullptr;
    cudnnCreateTensorDescriptor(&xDb);
    cudnnCreateTensorDescriptor(&dyDb);
    cudnnCreateTensorDescriptor(&dxDb);
    cudnnCreateFilterDescriptor(&wDb);
    cudnnCreateFilterDescriptor(&dwDb);
    cudnnCreateConvolutionDescriptor(&convDb);
    cudnnSetTensor4dDescriptor(xDb, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1,
                               4, 4);
    cudnnSetTensor4dDescriptor(dyDb, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1,
                               2, 2);
    cudnnSetTensor4dDescriptor(dxDb, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1,
                               4, 4);
    cudnnSetFilter4dDescriptor(wDb, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 1,
                               3, 3);
    cudnnSetFilter4dDescriptor(dwDb, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 1,
                               3, 3);
    cudnnSetConvolution2dDescriptor(convDb, 0, 0, 1, 1, 1, 1, 0,
                                    CUDNN_DATA_FLOAT);
    std::vector<float> Xb(16), Wb(9), dYb(4), dXb(16, 0.5f), dWb(9, 0.25f);
    for (auto &v : Xb) v = dist(rng);
    for (auto &v : Wb) v = dist(rng);
    for (auto &v : dYb) v = dist(rng);
    size_t ws = 999;
    EXPECT_TRUE(cudnnGetConvolutionBackwardDataWorkspaceSize(
                    dh, wDb, dyDb, convDb, dxDb,
                    CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, &ws) ==
                CUDNN_STATUS_SUCCESS);
    EXPECT_TRUE(ws == 0);
    EXPECT_TRUE(cudnnGetConvolutionBackwardDataWorkspaceSize(
                    dh, wDb, dyDb, convDb, dxDb,
                    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT, &ws) ==
                CUDNN_STATUS_NOT_SUPPORTED);
    const float al = 1.0f, be = 0.0f;
    std::vector<float> refDx(16, 0), refDw(9, 0);
    rouge_conv_bwd_data(1, 1, 4, 4, 1, 3, 3, 1, 0, dYb.data(), Wb.data(),
                        refDx.data());
    rouge_conv_bwd_filter(1, 1, 4, 4, 1, 3, 3, 1, 0, Xb.data(), dYb.data(),
                          refDw.data());
    EXPECT_TRUE(cudnnConvolutionBackwardData(
                    dh, &al, wDb, Wb.data(), dyDb, dYb.data(), convDb,
                    CUDNN_CONVOLUTION_BWD_DATA_ALGO_1, nullptr, 0, &be, dxDb,
                    dXb.data()) == CUDNN_STATUS_SUCCESS);
    EXPECT_TRUE(dXb == refDx);
    EXPECT_TRUE(cudnnGetConvolutionBackwardFilterWorkspaceSize(
                    dh, xDb, dyDb, convDb, dwDb,
                    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0, &ws) ==
                CUDNN_STATUS_SUCCESS);
    EXPECT_TRUE(ws == 0);
    EXPECT_TRUE(cudnnConvolutionBackwardFilter(
                    dh, &al, xDb, Xb.data(), dyDb, dYb.data(), convDb,
                    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1, nullptr, 0, &be, dwDb,
                    dWb.data()) == CUDNN_STATUS_SUCCESS);
    EXPECT_TRUE(dWb == refDw);
    cudnnDestroyTensorDescriptor(xDb);
    cudnnDestroyTensorDescriptor(dyDb);
    cudnnDestroyTensorDescriptor(dxDb);
    cudnnDestroyFilterDescriptor(wDb);
    cudnnDestroyFilterDescriptor(dwDb);
    cudnnDestroyConvolutionDescriptor(convDb);
  }
  EXPECT_TRUE(cudnnDestroy(dh) == CUDNN_STATUS_SUCCESS);

  // --- cublas L1/L2 поверх своих ядер ---
  {
    cublasHandle_t bh = nullptr;
    EXPECT_TRUE(cublasCreate_v2(&bh) == CUBLAS_STATUS_SUCCESS);
    // pointer mode: host ok, device — громкий отказ.
    {
      cublasPointerMode_t m = CUBLAS_POINTER_MODE_DEVICE;
      EXPECT_TRUE(cublasGetPointerMode_v2(bh, &m) == CUBLAS_STATUS_SUCCESS);
      EXPECT_TRUE(m == CUBLAS_POINTER_MODE_HOST);
      EXPECT_TRUE(cublasSetPointerMode_v2(bh, CUBLAS_POINTER_MODE_HOST) ==
                  CUBLAS_STATUS_SUCCESS);
      EXPECT_TRUE(cublasSetPointerMode_v2(bh, CUBLAS_POINTER_MODE_DEVICE) ==
                  CUBLAS_STATUS_NOT_SUPPORTED);
    }
    const int n = 64;
    std::vector<float> x(n), y(n), ys(n);
    std::vector<double> xd(n), yd(n);
    for (int i = 0; i < n; ++i) {
      x[i] = dist(rng);
      y[i] = dist(rng);
      ys[i] = y[i];
      xd[i] = dist(rng);
      yd[i] = dist(rng);
    }
    const float one = 1.0f, two = 2.0f, zero = 0.0f;
    EXPECT_TRUE(cublasSaxpy_v2(bh, n, &two, x.data(), 1, y.data(), 1) ==
                CUBLAS_STATUS_SUCCESS);
    for (int i = 0; i < n; ++i) EXPECT_TRUE(y[i] == 2.0f * x[i] + ys[i]);
    const double done = 1.0;
    EXPECT_TRUE(cublasDaxpy_v2(bh, n, &done, xd.data(), 1, yd.data(), 1) ==
                CUBLAS_STATUS_SUCCESS);
    float dot = 0;
    EXPECT_TRUE(cublasSdot_v2(bh, n, x.data(), 1, ys.data(), 1, &dot) ==
                CUBLAS_STATUS_SUCCESS);
    EXPECT_TRUE(dot == rouge_sdot(n, x.data(), 1, ys.data(), 1));
    double ddot = 0;
    EXPECT_TRUE(cublasDdot_v2(bh, n, xd.data(), 1, yd.data(), 1, &ddot) ==
                CUBLAS_STATUS_SUCCESS);
    EXPECT_TRUE(ddot == rouge_ddot(n, xd.data(), 1, yd.data(), 1));
    float nrm = 0;
    EXPECT_TRUE(cublasSnrm2_v2(bh, n, x.data(), 1, &nrm) ==
                CUBLAS_STATUS_SUCCESS);
    EXPECT_TRUE(nrm == rouge_snrm2(n, x.data(), 1));
    EXPECT_TRUE(cublasSscal_v2(bh, n, &two, y.data(), 1) ==
                CUBLAS_STATUS_SUCCESS);
    EXPECT_TRUE(cublasScopy_v2(bh, n, x.data(), 1, ys.data(), 1) ==
                CUBLAS_STATUS_SUCCESS);
    EXPECT_TRUE(ys == x);
    EXPECT_TRUE(cublasSswap_v2(bh, n, x.data(), 1, y.data(), 1) ==
                CUBLAS_STATUS_SUCCESS);
    // gemv N col-major 5x4 против прямой петли.
    {
      const int m = 5, k = 4;
      std::vector<float> A(m * k), xx(k), yy(m, 0.0f);
      for (auto &v : A) v = dist(rng);
      for (auto &v : xx) v = dist(rng);
      EXPECT_TRUE(cublasSgemv_v2(bh, CUBLAS_OP_N, m, k, &one, A.data(), m,
                                 xx.data(), 1, &zero, yy.data(), 1) ==
                  CUBLAS_STATUS_SUCCESS);
      for (int i = 0; i < m; ++i) {
        double s = 0;
        for (int j = 0; j < k; ++j) s += (double)A[j * m + i] * xx[j];
        EXPECT_TRUE(std::fabs((double)yy[i] - s) < 1e-6 * (1 + std::fabs(s)));
      }
      // T: A хранится k×m... точнее n×m при op m×n: A — 4x5 col-major.
      std::vector<float> At(k * m), yy2(m, 0.0f);
      for (auto &v : At) v = dist(rng);
      EXPECT_TRUE(cublasSgemv_v2(bh, CUBLAS_OP_T, m, k, &one, At.data(), k,
                                 xx.data(), 1, &zero, yy2.data(), 1) ==
                  CUBLAS_STATUS_SUCCESS);
      for (int i = 0; i < m; ++i) {
        double s = 0;
        for (int j = 0; j < k; ++j) s += (double)At[i * k + j] * xx[j];
        // op(A)[i][j] = At_stored[j][i] = At[i*ldb+j] при ldb=k. Да.
        EXPECT_TRUE(std::fabs((double)yy2[i] - s) < 1e-6 * (1 + std::fabs(s)));
      }
      // Комплексные: dotu/dotc + axpy smoke.
      std::vector<cuComplex> cx(n), cy(n);
      for (int i = 0; i < n; ++i) {
        cx[i].x = dist(rng);
        cx[i].y = dist(rng);
        cy[i].x = dist(rng);
        cy[i].y = dist(rng);
      }
      cuComplex r{};
      EXPECT_TRUE(cublasCdotu_v2(bh, n, cx.data(), 1, cy.data(), 1, &r) ==
                  CUBLAS_STATUS_SUCCESS);
      {
        double sr = 0, si = 0;
        for (int i = 0; i < n; ++i) {
          sr += (double)cx[i].x * cy[i].x - (double)cx[i].y * cy[i].y;
          si += (double)cx[i].x * cy[i].y + (double)cx[i].y * cy[i].x;
        }
        EXPECT_TRUE(std::fabs((double)r.x - sr) < 1e-5 * (1 + std::fabs(sr)));
        EXPECT_TRUE(std::fabs((double)r.y - si) < 1e-5 * (1 + std::fabs(si)));
      }
      EXPECT_TRUE(cublasCdotc_v2(bh, n, cx.data(), 1, cy.data(), 1, &r) ==
                  CUBLAS_STATUS_SUCCESS);
      {
        double sr = 0, si = 0;
        for (int i = 0; i < n; ++i) {
          sr += (double)cx[i].x * cy[i].x + (double)cx[i].y * cy[i].y;
          si += (double)cx[i].x * cy[i].y - (double)cx[i].y * cy[i].x;
        }
        // dotc: sum conj(x)*y = (xr*yr+xi*yi, xr*yi-xi*yr). Да.
        EXPECT_TRUE(std::fabs((double)r.x - sr) < 1e-5 * (1 + std::fabs(sr)));
        EXPECT_TRUE(std::fabs((double)r.y - si) < 1e-5 * (1 + std::fabs(si)));
      }
      float cn = 0;
      EXPECT_TRUE(cublasCnrm2_v2(bh, n, cx.data(), 1, &cn) ==
                  CUBLAS_STATUS_SUCCESS);
      EXPECT_TRUE(cn > 0);
    }
    EXPECT_TRUE(cublasDestroy_v2(bh) == CUBLAS_STATUS_SUCCESS);
  }

  if (failures == 0) std::printf("PASS: cublas + cudnn shims\n");
  return failures ? 1 : 0;
}
